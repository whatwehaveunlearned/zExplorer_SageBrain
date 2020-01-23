#!/usr/bin/python

#server sockets
import asyncio
import websockets

#Server stuff
import zerorpc
import logging
logging.basicConfig()
#Data Science Imports
import pandas as pd
#Debugg
import pdb
#json parser
import json
#To Import from Arxiv
import arxiv

#Import Brain Classes
from classes.session_class import Session
from classes.document_class import Document
from classes.topic_class import Topic #Not using this now
from classes.zotero_class import Zotero

#Interface Class to talk with Node
class SageBrain(object):
    """ Sage Brain Class"""
    def __init__(self, session_id):
        self.id = "brainInterface"
        self.sessions = []
        self.session_counter = -1
        self.actualSession = -1
        self.addSession(session_id)
        self.zotero = Zotero('2476068','user','ravDnfy0bMKyuDrKq5kNz5Rh')
        self.sess_folder = './sessData/' + session_id
        self.globalSess = Session('globalSess')

    def Zotero(self,function_name,collection_key,itemKeys,collection_items):
        value_to_return = False
        if function_name == 'getCollections':
            value_to_return = self.zotero.getColletions()
        elif function_name == 'getCollectionItems':
            value_to_return = self.zotero.getCollectionItems(collection_key)
        elif function_name == 'downloadItems':
            value_to_return = self.zotero.downloadItems(itemKeys,collection_items,self.sess_folder)
        
        return value_to_return

    def DocInterface(self, fileName, doc_id,dataType,metadata):
        """ Brain Interface """
        if dataType == 'zoteroCollection':
            for index,each_doc in enumerate(metadata):
                doc_in_sess = self.actualSession.docInSess(metadata[index]['key'])
                doc_in_global = self.globalSess.docInSess(metadata[index]['key'])
                if doc_in_sess == True:
                    print ("Doc in sess")
                elif doc_in_global == False:
                    print ("New Document")
                    doc = Document(self.actualSession, metadata[index]['name'],  metadata[index]['key'], 'zotero', "user", each_doc)
                    self.globalSess.addDoc(doc)
                    self.actualSession.addDoc(doc)
                else:
                    print ("Doc in global")
                    doc_from_global = self.globalSess.returnDoc(metadata[index]['key'])
                    doc = Document(self.actualSession, 'name',  'key', 'inSession', "user", doc_from_global)
                    self.actualSession.addDoc(doc)
            #We get the topic and words Using Umap NSA algorithm and we include them into session
            self.actualSession.get_topics(self.actualSession.documents)
        else:
            doc = Document(self.actualSession, fileName, doc_id, "doc", "user", False)
            self.actualSession.addDoc(doc)
        #Get Umap fit
        self.get_projections()
        current_data = self.send_current()
        # pdb.set_trace()
        return current_data

    def search_Arxiv(self,paperData,topicData):
        # server.actualSession.search_arxiv(paperData,topicData)
        
        #store using ID (How do they mach the Zotero Ones?)
        def custom_slugify(obj):
            try:
                return obj.get('id').split('/')[-1].split('.')[0] + obj.get('id').split('/')[-1].split('.')[1]
            except IndexError:
                pdb.set_trace()
        #Initialization
        number_of_papers = 0
        query = []
        #parameters
        Number_of_results = 3
        max_results_per_query = 50
        #I keep the unordered list since D3 will redraw elements in order and would reposition elements in the return if not but I need the ordered list to choose the proper topic
        paper_data_ordered_by_weight = paperData.sort_values(by=['weight'], ascending=False)
        # topic_data_ordered_by_weight = topicData.sort_values(by=['weight'], ascending=False)
        pdb.set_trace()
        #papers queries
        print(paper_data_ordered_by_weight['title'].iloc[0])
        query.append(paper_data_ordered_by_weight['title'].iloc[0])
        # query.append(paperData['title'][1])
        # query.append(paperData['title'][2])
        #topics queries
        # query.append(' '.join([word['word'] for word in topicData['words'][0]])) 
        # query.append(' '.join([word['word'] for word in topicData['words'][1]])) 
        # query.append(' '.join([word['word'] for word in topicData['words'][2]])) 
        for each_query in query:
            #Query Iterator
            result = arxiv.query(query=each_query,max_results=max_results_per_query,iterative=True)
            for paper in result():
                #We dont want more results than the number of results
                if number_of_papers == Number_of_results:
                    break
                else:
                    # pdb.set_trace()
                    # pdb.set_trace()
                    # #we always save in global in case for other sessions
                    # self.globalSess.addDoc(doc)
                    #Check if paper in Session first by ID
                    doc_in_sess = self.actualSession.docInSess(custom_slugify(paper))
                    #Since different IDs in case I check the name too
                    title_in_sess = self.actualSession.documents['title'].isin([paper['title']]).any()
                    #Check if paper in Global first by ID
                    doc_in_global = self.globalSess.docInSess(custom_slugify(paper))
                    #Since different IDs in case I check the name too
                    title_in_global = self.globalSess.documents['title'].isin([paper['title']]).any()
                    #IF paper is not in this session
                    if(doc_in_sess == False and title_in_sess==False):
                        #Check if its in Global by ID or Title if found we break
                        if(doc_in_global or title_in_global):
                            if(doc_in_global):
                                print("Document from Arxiv already in Global")
                                doc_from_global = self.globalSess.returnDoc(custom_slugify(paper))
                                doc = Document(self.actualSession, 'name',  'key', 'inSession', "user", doc_from_global)
                                # self.actualSession.addDoc(doc)
                            elif(title_in_global):
                                #Check how I can get the paper from global by title if I end up here
                                pdb.set_trace()
                        #Paper is new not in GLOBAL need to download and process
                        else:
                            #Download Text
                            print('##### ' + paper['title'] + ' ######')
                            try:
                                arxiv.custom_download(paper, dirpath= self.sess_folder + '/documents/', slugify=custom_slugify)
                            except TypeError:
                                break
                            doc = Document(self, paper['title'],  custom_slugify(paper), 'arxiv', "user", paper)
                            self.globalSess.addDoc(doc);
                        #If the text or topics vector or the textis smaller than the smallest processed we ignore that document and dont add it to the session.
                        if self.actualSession.topic_min_length <= len(doc.topics_vector):
                            if self.actualSession.text_min_length <= len(doc.abstract_vector):
                                number_of_papers = number_of_papers + 1
                                self.actualSession.addDoc(doc)
                            else:
                               print('Paper can not be imported topic vector smaller than smallest vector in system')
                        else:
                            print('Paper can not be imported topic vector smaller than smallest vector in system')
                    else:
                        #Paper already in session
                        print('### ' + paper['title'] + ' already in Session ##')
        
        
        #We store the new papers in the session and global
        self.globalSess.storeSessData()
        self.actualSession.storeSessData()
        #We get the topic and words Using Umap NSA algorithm and we include them into session
        self.actualSession.get_topics(self.actualSession.documents)
        #Get Umap fit
        self.get_projections()
        current_data = self.send_current()
        # pdb.set_trace()
        return current_data

    def UpdateModel(self,new_paper_data):
        self.actualSession.update_model(new_paper_data)
        return self.actualSession.documents

        
    def addCitations(self):
        documents_msg = []
        for each_doc in self.actualSession.documents:
            print (each_doc.title)
            each_doc.GetScholarInfo()
            documents_msg.append(each_doc.create_document_msg())
        # pdb.set_trace()
        return {"documents":documents_msg}

    def get_projections(self):
        #We get the projections of the papers in the session
        self.actualSession.documents = self.actualSession.train_fit_UMAP_2D(self.actualSession.documents)
        self.globalSess.storeSessData()
        self.actualSession.storeSessData()


    def addSession(self, session_id):
        """ Function to add new sessions """
        self.session_counter = self.session_counter + 1
        self.sessions.append(Session(session_id))  
        self.actualSession = self.sessions[self.session_counter]

    def send_current(self):
        #Assign Session Papers to Session Topics
        self.actualSession.assign_topics_to_documents()
        #We get the years
        years = self.actualSession.get_years()
        pdb.set_trace()
        return {"documents":self.actualSession.documents,"years":years,"authors":self.actualSession.authorList,"doc_topics":{'topics':self.actualSession.topics, 'topic_params':self.actualSession.topic_params, 'order':self.actualSession.topics.columns.values.tolist(), 'words':self.actualSession.words.to_json(orient='records')}}
        # return {"documents":self.actualSession.documents.to_json(),"years":years.to_json(),"authors":self.actualSession.authorList.to_json(orient='records'),"doc_topics":{'topics':self.actualSession.topics.to_json(), 'order':json.dumps(self.actualSession.topics.columns.values.tolist()), 'words':json.dumps(self.actualSession.words.to_json(orient='records'))}}
        # return {"documents":self.actualSession.documents.to_json(),"years":years.to_json(),"authors":"self.actualSession.authorList.to_json()","doc_topics":{'topics':self.actualSession.topics.to_json(), 'order':json.dumps(self.actualSession.topics.columns.values.tolist()), 'words':json.dumps(self.actualSession.words.to_json())}}
        

async def handler(websocket,path):
    print('Server Running ...')
    data_zotero = server.Zotero('getCollections',False,False,False)
    await websocket.send(json.dumps({
        "message" : data_zotero,
        "type" : "collections"
    }))
    consumer_task = asyncio.ensure_future(handle_message(websocket,path,server))
    done, pending = await asyncio.wait(
        [consumer_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
            task.cancel()

async def handle_message(websocket, path,server):
    while True:
        message = await websocket.recv()
        json_message = json.loads(message)
        if json_message['type']=='collections':
            data_zotero = server.Zotero("getCollectionItems",json_message['msg'],False,False)
            collection_items = json.loads(data_zotero)
            pdf_ids = []
            for  i in range(0,len(collection_items)):
                pdf_ids.append(collection_items[i]['pdf_file'])
            await websocket.send(json.dumps({
                "message" : data_zotero,
                "type" : "collection_items"
            }))
        elif json_message['type']=='add_papers':
            data_zotero = server.Zotero("downloadItems",False,pdf_ids,collection_items)
            message = server.DocInterface(False,'0','zoteroCollection',collection_items)
            await websocket.send(json.dumps({
                "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':message['doc_topics']['topics'].to_json(),'topic_params':message['doc_topics']['topic_params'].to_json(), 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
                # "message" : message,
                "type" : "sageBrain_data"
            }))
        elif json_message['type']=='search_arxiv':
            new_paper_data = pd.DataFrame(json_message['msg']['papers'])
            new_topic_data = pd.DataFrame(json_message['msg']['topics'])
            message = server.search_Arxiv(new_paper_data,new_topic_data)
            await websocket.send(json.dumps({
                "message": {'documents':message['documents'].to_json(),'doc_topics':{'topics':message['doc_topics']['topics'].to_json(),'topic_params':message['doc_topics']['topic_params'].to_json(), 'order': message['doc_topics']['order'], 'words':message['doc_topics']['words']},'years':message['years'].to_json(),'authors':message['authors'].to_json()},
                # "message" : message,
                "type" : "sageBrain_data"
            }))
        elif json_message['type']=='update_model':
            # pdb.set_trace()
            #TOPICS AND WORDS DONT CHANGE SO WE JUST PASS THEM BACK
            new_paper_data = pd.DataFrame(json_message['msg']['papers'])
            # pdb.set_trace()
            documents = server.UpdateModel(new_paper_data)
            await websocket.send(json.dumps({
                "message": {'documents':documents.to_json()},
                # "message" : message,
                "type" : "update_model"
            }))

def main():
    start_server = websockets.serve(handler, "0.0.0.0", 3000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

server = SageBrain('sess3')

#start process
if __name__ == "__main__":
    main()