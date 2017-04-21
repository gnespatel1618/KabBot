import sys
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random
import requests
import gmaps
from gmaps import Geocoding
from random import randint
import sklearn.model_selection as m_sel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from uber_rides.session import Session
from uber_rides.client import UberRidesClient
import uber_rides.errors as uber_error
from nltk.wsd import lesk
from nltk.tag import brill, brill_trainer
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.data import load
from nltk.corpus.reader import TaggedCorpusReader


session = Session(server_token='gQqt-C_l3O2CSZ0Y2TNjIRJQmSmowvcJPMAOBLoB')
client = UberRidesClient(session)

class KabBot:
    
    def __init__(self):
        self.SessionInfo = {
            "source": None,
            "destination": None,
            "intent": None
        }
        self.nb_naive, self.vect = self.train_intent_mapper()
        self.brill_tagger = self.get_brill_tagger()
        
    def get_brill_tagger(self):
        train_data = TaggedCorpusReader('.', 'tagged_input_sentences.txt', sep="/")
        traindata= list(train_data.tagged_sents())
        postag= load('taggers/maxent_treebank_pos_tagger/english.pickle')
        templates = [
                brill.Template(brill.Pos([-1])),
                brill.Template(brill.Pos([1])),
                brill.Template(brill.Pos([-2])),
                brill.Template(brill.Pos([2])),
                brill.Template(brill.Pos([-2, -1])),
                brill.Template(brill.Pos([1, 2])),
                brill.Template(brill.Pos([-3, -2, -1])),
                brill.Template(brill.Pos([1, 2, 3])),
                brill.Template(brill.Pos([-1]), brill.Pos([1])),
                brill.Template(brill.Word([-1])),
                brill.Template(brill.Word([1])),
                brill.Template(brill.Word([-2])),
                brill.Template(brill.Word([2])),
                brill.Template(brill.Word([-2, -1])),
                brill.Template(brill.Word([1, 2])),
                brill.Template(brill.Word([-3, -2, -1])),
                brill.Template(brill.Word([1, 2, 3])),
                brill.Template(brill.Word([-1]), brill.Word([1]))]        
        trainer = BrillTaggerTrainer(postag, templates = templates, trace = 3)
        brill_tagger = trainer.train(traindata, max_rules = 10)
        return brill_tagger
    
    def train_intent_mapper(self):
        cab_bot_data_df = pd.read_csv('cab_bot_data.csv')
        cab_bot_data_df['Category_label'] = cab_bot_data_df.Category.map({'Greetings':1, 
                                            'Look':2, 'Book':3, 'Fare_Estimation':4,'Cancel':5, 'Duration':6})
        X = cab_bot_data_df.Questions
        y = cab_bot_data_df.Category_label
        X_train,X_test,Y_train,Y_test = m_sel.train_test_split(X,y,test_size=0.30,random_state=30)
        vect = CountVectorizer()
        vect.fit(X_train)        
        X_train_dtm = vect.transform(X_train)
        nb = MultinomialNB()
        nb.fit(X_train_dtm, Y_train)
        return (nb, vect)
    
    def find_intent(self, text):
        category = ['Greetings','Look','Book','Fare_Estimation',
                    'Cancel','Duration']
        txt_l = [text]
        vect_text = self.vect.transform(txt_l)
        intent = None
        if len(list(vect_text.data)) != 0:
            predict_value = self.nb_naive.predict(vect_text)
            predict_val = int(predict_value)
            intent = category[predict_val-1].lower()
        return intent
    
    def extract_location(self, inp):
        tagged = self.brill_tagger.tag(word_tokenize(inp))
        chunkGram = """Source: {<IN>(<NN.*><,>?)+}"""
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Source'):
            self.SessionInfo["source"] = ' '.join(list(zip(*subtree))[0][1:]) 

        chunkGram = """Destination: {<TO>(<NN.*><,>?)+}"""
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Destination'):
            self.SessionInfo["destination"] = ' '.join(list(zip(*subtree))[0][1:]) 
        
        return self.SessionInfo["source"], self.SessionInfo["destination"]
        
    def storing_info(self, inp, intent):
        if intent == 'book' or intent == 'look':
            self.extract_location(inp)
            cab_types = ['uberpool','ubergo','uberx', 'uberxl']
            self.cab_types = 'ubergo'
            for cab in cab_types: 
                if cab in inp: self.cab_types = inp
        
    def handle_require_field(self, intent):
        requiredKey = ['source','destination']
        notAvail = []
        resp = ''
        boolean = True
        for key in requiredKey:
            if self.SessionInfo[key] == None: 
                notAvail.append(key)
        # print(notAvail,len(notAvail))        
        if(len(notAvail)!=0):
            boolean = False
            resp = 'Please provide all the necessary details'
            if(len(notAvail)==2):
                resp = 'You have not provided any source and destination.'
        return resp, boolean
    
    def cab_details(self):
        driver_name = random.choice(["Driver1", "Sharmaji", "Chachaji", "Chunnu", "Munni", "beta", "daadaji", "daadi", "bhai", "shinde"])
        cab_no = 'KA-' + str(random_with_N_digits(2)) + '-AT-' + str(random_with_N_digits(4))
        driver_rating = str(random.choice([3.5, 3.8, 4.0, 4.2, 4.4, 4.5, 4.8, 5]))
        mobile_number = random.choice(['8876','9888','8054','7077']) + str(random_with_N_digits(6))
        return 'Uber driver '+ driver_name + ' with mobile no. '+ mobile_number + ' arriving on cab ' + cab_no + ' has rating of '+ driver_rating + ' stars.'

    def look_uber(self, source, destination, cab_type, intent):
        api = Geocoding()
        try:
            source_loc = api.geocode(source)
            source_lat_long = source_loc[0]['geometry']['location']
            start_lat=source_lat_long['lat']
            start_lng=source_lat_long['lng']
        except:
            return 'Enter Valid Source.\n'

        try:
            destination_loc = api.geocode(destination)
            destination_lat_long = destination_loc[0]['geometry']['location']
            end_lat=destination_lat_long['lat']
            end_lng=destination_lat_long['lng']
        except:
            return 'Enter Valid destination.\n'

        try:
            response = client.get_price_estimates(
                start_latitude=start_lat,
                start_longitude=start_lng,
                end_latitude=end_lat,
                end_longitude=end_lng,
                seat_count=2
                )    

            estimate = response.json 

            for cab_types in estimate['prices']:
                if cab_types['localized_display_name'].lower() == cab_type.lower():
                    if (intent == 'book'):
                        out = 'Booking ' + (cab_type) + ' with averege fare ' + (cab_types['estimate']) + '. Your journy will be ' + str(cab_types['distance']) + ' KM long and will take ' + str(cab_types['duration']/60) + ' minutes. '
                        return out + self.cab_details()
                    else:
                        out = str(cab_type) + ' with averege fare ' + (cab_types['estimate']) + 'is available. Distance will be ' + str(cab_types['distance']) + ' KM and it will take ' + str(cab_types['duration']/60) + ' minutes.'
                        print(out)
                        return str(out)
        except uber_error.ClientError as e:
            return 'Distance between two points exceeds 100 miles'
    
    def generating_response(self, inp, intent):
        rep, flag = self.handle_require_field(intent)
        if flag == False:
            return rep
        
        resp = rep
        if intent == 'look' :    
            resp = resp + 'Looking for a cab from {}'.format(self.SessionInfo.get('source',''),'to '+ self.SessionInfo['destination'] if self.SessionInfo['destination']!=None else '' )
            resp = resp + '\n' + self.look_uber(self.SessionInfo['source'], self.SessionInfo['destination'], self.cab_types, self.SessionInfo['intent'])
            
        elif intent == 'book':
            resp = resp + 'Looking for a cab from {} {}'.format(self.SessionInfo.get('source',''),'to '+self.SessionInfo['destination'] if self.SessionInfo['destination']!=None else '' )
            #resp = resp + '\n' + self.look_uber(self, self.SessionInfo['source'],self.SessionInfo['destination'])
            resp = resp + '\n' + self.look_uber(self.SessionInfo['source'], self.SessionInfo['destination'], self.cab_types, self.SessionInfo['intent'])       

        '''elif intent == 'book' and self.booked == True :
            resp = 'You have already booked a cab with booking id '+ str(self.SessionInfo['bookingid'])
            resp = resp + '\n' + 'We dont provide multiple booking'
        '''
        return resp 
    
    def respond(self, query):
        inp = query.lower()
        intent = self.find_intent(inp)
        #print(intent)
        self.SessionInfo["intent"] = intent if intent else self.SessionInfo["intent"]
        if intent:
            self.storing_info(inp, intent)
        else: 
            for key in ['source', 'destination']:
                if self.SessionInfo[key] == None:
                    #print(key)
                    self.SessionInfo[key] = inp
        response = self.generating_response(inp, intent)
        return response
        