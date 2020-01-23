# -*- coding: utf-8 -*-
import pandas as pd
import pdb
import numpy

#Read Pdf imports
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
#To acccess table of contents
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
#Date Parsing
from dateutil.parser import parse as dateParser

#Functions from NSA Are called from shell because they run in Python3
import subprocess
import os

#for parsing the sections
import re
#Function to do bert Need to have bert_serving_server on from terminal: bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=4 
from bert_serving.client import BertClient
bc = BertClient() 
import spacy
nlp = spacy.load('en')

from topic_extractor import topic_extractor

#Document Class
class Document:
    """ Document Class """
    def __init__(self, session, name, doc_id, doc_type, doc_user, metadata):
        #Initialize doc parameters
        self.title = False
        self.pdf_file = False
        self.parent_item = False
        self.tags = False
        self.url = False
        self.year = False
        self.citations = False
        self.versions = False
        self.clusterID = False
        self.citations_list = False
        self.notes = False
        self.type = False
        self.globalID = False
        self.conference = False
        self.organization = False
        self.pages = False
        self.text = False
        self.abstract = False
        self.conclusion = False
        self.citationArticles = False
        self.authors = False
        self.topics =  False
        self.topic_params = False
        self.words = False
        self.session = session
        self.folder = self.session.sess_folder
        self.metadata = metadata
        self.bert_vector= False
        self.title_vector = False
        self.abstract_vector = False
        self.conclusion_vector = False
        self.topics_vector = False
        self.toc = False
        #this are filled up after we process the current session  in Session class asign_topics_to_documents
        self.assigned_sess_topics = []
        self.weight = 0
        # self.vector_2D = False;
        if doc_type == 'zotero':
            if metadata['pdf_file'] == None:
                print ('No File')
                self.text = 'No File'
            #Parse Metadata
            self.parseMetadata()
            print(self.title)
            try:
                self.text = self.get_text(metadata['pdf_file'] + '.pdf',self.folder).lower()
                #Calculate document models
                self.calculate_doc_models()
            except TypeError:
                print ('Parsing Error')
                self.text = 'Parsing Error'
                self.topics = 'Parsing Error'
        elif doc_type == 'arxiv':
            self.parseArxiv(doc_id)
            print(self.title)
            # pdb.set_trace()
            self.text = self.get_text(self.pdf_file + '.pdf',self.folder).lower()
            #Calculate document models
            self.calculate_doc_models()
        elif doc_type == 'inSession':
            self.parseFromSession()
            print(self.title)
        else:
            self.metadata = False
            self.title = name.split('.')[0] #Delete the extension
            self.id = doc_id
            self.type = doc_type
            self.user = doc_user
            self.text = self.get_text(name,self.folder).lower()
            self.toc = self.get_toc(self.folder + name)
            if self.toc:
                self.sections = self.get_sections()
            else:
                self.sections = False

    def calculate_bert(self,text):
        # pdb.set_trace()
        spacey_doc = nlp(text)
        vector_sentences_list = []
        for sentence in spacey_doc.sents:
            #trim small sentences for speed
            if len(sentence)>5:
                vector_sentences_list.extend(bc.encode([str(sentence)])[0].tolist())
        # pdb.set_trace()
        return vector_sentences_list
    
    def topic_vector(self):
        topics = self.topics['topics'].columns.values.tolist()
        topics_average_vector = []
        for each_topic in topics:
            this_topic_vector = []
            this_topic_words = self.topics['topics'][each_topic].dropna().values.tolist()
            for word in this_topic_words:
                if word[len(word)-1]=='*':
                    this_topic_vector.append(self.words[self.words['word*']==word]['vector'].values.tolist()[0])
            #After geting the vector per word we avg them to get a single vector per topic
            topics_average_vector.extend(numpy.mean(this_topic_vector,axis=0))
        # return numpy.mean(topic_average_vector, axis=0)
        return topics_average_vector

    def calculate_doc_models(self):
        # pdb.set_trace()
        #doc structure
        # pdb.set_trace()
        self.toc = self.get_toc(self.folder + '/documents/' + self.pdf_file +'.pdf') #If I have toc I can loop to get the different sections (for latter)
        #doc abstract text
        self.abstract = self.text[self.text.find('abstract'):self.text.find('introduction')]
        if(len(self.abstract)==0):
            self.abstract = self.text[self.text.find('abstract'):self.text.find('keywords')]
        #find last ocurrence of references
        references_ocurrences = [m.start() for m in re.finditer('references',self.text)]
        #Get conclusion if possible
        if(len(references_ocurrences)>0):
            last_reference_ocurrence = references_ocurrences[len(references_ocurrences)-1]
            self.conclusion = self.text[self.text.find('conclusion'):last_reference_ocurrence]
            if len(self.conclusion) == 0:
                self.conclusion = self.text[self.text.find('conclusions'):last_reference_ocurrence]
        # #Text vector Too big full text
        # self.bert_vector = bc.encode([self.text])[0].tolist()
        #Title vector
        self.title_vector = bc.encode([self.title])[0]
        #Abstract Vector
        if(len(self.abstract)>0 and len(self.abstract)<5000):
            self.abstract_vector = self.calculate_bert(self.abstract)
        else:
            print  ('Error parsing abstract is empty or wrongly parsed')
            self.abstract = self.text[:1500]
            self.abstract_vector = self.calculate_bert(self.abstract)
            # pdb.set_trace()
        #Conclusion Vector
        # if(len(self.conclusion)>0):
        #     self.conclusion_vector = self.calculate_bert(self.conclusion)
        # else:
        #     print  ('Error parsing conclusion is empty')
        #     self.conclusion_vector = False
        #     # pdb.set_trace()
        #Topics
        self.topics = self.get_topics()
        #Create topic vector
        self.topics_vector = self.topic_vector()
        # pdb.set_trace()

    def get_topics(self):
        """Function to calculate the topics from a document, returns the topics in order of importance and the words"""
        doc_dictionary = self.create_document_msg()
        doc_dictionary = pd.DataFrame.from_dict([doc_dictionary])
        topics_data = topic_extractor(doc_dictionary,'document')
        self.topics = topics_data['topics']
        self.topic_params = topics_data['topic_params']
        self.words = topics_data['lvls_df']
        return {'topics':self.topics,'words':self.words,'topic_params':self.topic_params}
 
    #Parse from Session
    def parseFromSession(self):
        self.title = self.metadata.title
        self.pdf_file = self.metadata.pdf_file
        self.parent_item = self.metadata.parent_item
        self.tags = self.metadata.tags
        self.url = self.metadata.url
        self.year = self.metadata.year
        self.citations = self.metadata.citations
        self.versions = self.metadata.versions
        self.clusterID = self.metadata.clusterID
        self.citations_list = self.metadata.citationsList
        self.notes = self.metadata.notes
        self.abstract = self.metadata.abstract
        self.conclusion = self.metadata.conclusion
        self.type = self.metadata.type
        self.globalID = self.metadata.globalID
        self.organization = self.metadata.organization
        self.pages = self.metadata.pages
        self.text = self.metadata.text.lower()
        self.citationArticles = self.metadata.citationArticles
        self.bert_vector = self.metadata.bert_vector
        self.abstract_vector = self.metadata.abstract_vector
        self.conclusion_vector = self.metadata.conclusion_vector
        self.title_vector = self.metadata.title_vector
        self.topics_vector = self.metadata.topics_vector
        self.topic_params = self.metadata.topic_params
        #this are filled up after we process the current session  in Session class asign_topics_to_documents
        self.assigned_sess_topics = []
        self.weight = 0
        parsed_authors = self.metadata.author
        self.authors = parsed_authors
        if (self.metadata.topics != 'False') and (self.metadata.topics != 'Parsing Error'):
            self.topics = {'topics':pd.read_json(self.metadata.topics),'words':pd.read_json(self.metadata.words)}
        else:
            self.topics = False
            self.words = False

    def parseMetadata(self): #I have 2 parent items and 2 notes, 2 tags I have to put them together in the near future only reading x now
        self.title = self.metadata["name"]
        self.title_vector = bc.encode([self.title])[0].tolist()
        self.pdf_file = self.metadata['pdf_file']
        self.parent_item = self.metadata['parentItem_x']
        self.tags = self.metadata['tags_x']
        self.url = self.metadata["url"]
        if self.metadata["date"]:
            self.year = dateParser(self.metadata["date"]).year
        else:
            self.year = False
        self.citations = False
        self.versions = False
        self.clusterID = False
        self.citations_list = False
        self.notes = self.metadata["note_x"]
        self.authors = self.metadata["creators"]
        self.abstract = False
        self.type = self.metadata["itemType"]
        # pdb.set_trace()
        self.globalID = self.metadata['key']
        # self.conference = self.metadata["publicationTitle"]
        self.organization = False
        self.pages = False
        self.citationArticles = False

    def parseArxiv(self,doc_id):
        self.title = self.metadata["title"]
        self.title_vector = bc.encode([self.title])[0].tolist()
        self.pdf_file = doc_id
        self.parent_item = False
        self.tags = self.metadata['tags']
        self.url = self.metadata["arxiv_url"]
        if self.metadata["published"]:
            self.year = dateParser(self.metadata["published"]).year
        else:
            self.year = False
        self.citations = False
        self.versions = False
        self.clusterID = False
        self.citations_list = False
        self.notes = False
        authors_paper = self.metadata["authors"]
        self.authors = []
        for each in authors_paper:
            self.authors.append({
                "firstName": each.split(' ')[0],
                "lastName": ' '.join(each.split(' ')[1:])
            })
        self.abstract = self.metadata['summary']
        self.type = False
        # pdb.set_trace()
        self.globalID = doc_id
        # self.conference = self.metadata["publicationTitle"]
        self.organization = self.metadata['affiliation']
        self.pages = False
        self.citationArticles = False

    def get_sections(self):
        sections = []
        temp_text = self.text
        for section in self.toc:
            if len(temp_text.split(section[1].upper())) > 0:
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1].upper())[0]})
                if len(temp_text.split(section[1].upper())) > 1:
                    temp_text = temp_text.split(section[1].upper())[1]
            elif len(temp_text.split(section[1].title())) > 0:
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1].title())[0]})
                if len(temp_text.split(section[1].title())) > 1:
                    temp_text = temp_text.split(section[1].title())[1]
            elif len(temp_text.split(section[1]) > 0):
                sections.append({"section":section[1],"id":section[0],"text":temp_text.split(section[1])[0]})
                if len(temp_text.split(section[1]) > 1):
                    temp_text = temp_text.split(section[1])[1]
            else:
                sections.append("Could not Parse")
        return sections

    def get_text(self,fname,folder, pages=None):
        text = False
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
        with open(folder + '/documents/' + fname, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()

        # close open handles
        converter.close()
        fake_file_handle.close()
        return text;

    def get_toc(self, pdf_path):
        infile = open(pdf_path, 'rb')
        parser = PDFParser(infile)
        document = PDFDocument(parser)
        toc = list()
        try:
            for (level, title, dest, a, structelem) in document.get_outlines():
                toc.append((level, title))
            return toc
        except Exception:
            return False

    """This is the function used to extract the document information and add it to the session.
    Is Important to add any new variables that are needed in the session here if we want to
     store the value"""
    def create_document_msg(self):
        #transform author dataframe into json to searialize
        # pdb.set_trace()
        if (self.topics != False) and (self.topics != 'Parsing Error'):
            topics = self.topics['topics'].to_json()
            words = self.topics['words'].to_json()
        else:
            topics = False
            words = False
        # if (isinstance(self.authors, pd.DataFrame)):
        #     authors = self.authors.to_json()
        # else:
        #     authors = False
        msg = {
            'user':'self.user',
            'text': self.text,
            'title' : self.title,
            'abstract':self.abstract,
            'conclusion':self.conclusion,
            'topics':topics,
            'words':words,
            'pdf_file': self.pdf_file,
            'parent_item': self.parent_item,
            'tags': self.tags,
            'url': self.url,
            'year': self.year,
            'citations': self.citations,
            'versions': self.versions,
            'clusterID': self.clusterID,
            'citationsList':self.citations_list,
            'notes': self.notes,
            'author': self.authors,
            'type':self.type,
            'globalID':self.globalID,
            # 'conference':self.conference,
            'organization': self.organization,
            'pages':self.pages,
            'citationArticles': self.citationArticles,
            'bert_vector':self.bert_vector,
            'abstract_vector':self.abstract_vector,
            'conclusion_vector': self.conclusion_vector,
            'title_vector': self.title_vector,
            'topics_vector':self.topics_vector,
            'topic_params':self.topic_params,
            'assigned_sess_topics' : self.assigned_sess_topics,
            'weight' : self.weight
        }
        return msg