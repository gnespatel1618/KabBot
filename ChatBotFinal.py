
# coding: utf-8

# ## ChatBot

# # Import Modules #

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


session = Session(server_token='gQqt-C_l3O2CSZ0Y2TNjIRJQmSmowvcJPMAOBLoB')
client = UberRidesClient(session)


# # Random No Generation #

# In[10]:

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


# # Brill Tagger #

# In[11]:

from nltk.wsd import lesk
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
import tkinter
from nltk.tag import brill, brill_trainer
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.data import load
from nltk.corpus.reader import TaggedCorpusReader


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


# # Source and Destination Extraction From Sentence # 

# In[12]:

def extract_location(inp):
    tagged = brill_tagger.tag(word_tokenize(inp))
    source = None
    destination = None
    chunkGram = """Source: {<IN>(<NN.*><,>?)+}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Source'):
        source = ' '.join(list(zip(*subtree))[0][1:]) 
    
    chunkGram = """Destination: {<TO>(<NN.*><,>?)+}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Destination'):
        destination = ' '.join(list(zip(*subtree))[0][1:]) 
    return source, destination


# # Training Intent Mapper #

# In[17]:

def train_intent_mapper():
    cab_bot_data_df=pd.read_csv('cab_bot_data.csv')
    cab_bot_data_df['Category_label'] = cab_bot_data_df.Category.map({'Greetings':1, 
                                        'Look':2, 'Book':3, 'Fare_Estimation':4,
                                        'Schedule':5, 'Cancel':6, 'Payment_Mode':7, 
                                        'Duration':8})
    X = cab_bot_data_df.Questions
    y = cab_bot_data_df.Category_label
    X_train,X_test,Y_train,Y_test = m_sel.train_test_split(X,y,test_size=0.30,random_state=30)
    vect = CountVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, Y_train)
    return (nb,vect)


# # Function For Finding Intent Of a Question #

# In[240]:

def find_intent(text,multinomial_nb,vect):
    category = ['Greetings','Look','Book','Fare_Estimation',
                'Schedule','Cancel','Payment_Mode','Duration']
    txt_l=[text]
    vect_text = vect.transform(txt_l)
    predict_value = multinomial_nb.predict(vect_text)
    predict_val=int(predict_value)
    intent = category[predict_val-1]
    return intent.lower() 


# # Function For Cab Details As Data Frame #

# In[14]:

def uber_cab_details(cab_dict):
    columns = ['Car type', 'Availability', 'Distance(Km)', 'Minimum Price', 'Maximum Price', 'ETA(min)' ]
    index=list(range(len(cab_dict['prices'])))
    df = pd.DataFrame(index=index, columns=columns)
    for i in index:
        df.loc[i,'Car type'] = cab_dict['prices'][i]['display_name']
        df.loc[i,'Availability'] = "Yes"
        df.loc[i,'Distance(Km)'] = cab_dict['prices'][i]['distance']
        df.loc[i,'Minimum Price'] = cab_dict['prices'][i]['low_estimate']
        df.loc[i,'Maximum Price'] = cab_dict['prices'][i]['high_estimate']
        df.loc[i,'ETA(min)'] = int(cab_dict['prices'][i]['duration']/100)
    df['Car type']= df['Car type'].str.lower()    
    return df      


# # Function For Cab Booking #

# In[26]:

def book_uber(source, destination, cabtype, noofseats,response_avail_df, SessInfo):
    df = response_avail_df
    driver_name=["Tharoon Veerasethu","Sharmaji","Chachaji","Chunnu","Munni","beta","daadaji","daadi","bhai","shinde","langoor"]
    for i in range(df.shape[0]):
        x=list(df.loc[i])
        if cabtype in x:
            for j in x:
                if j == cabtype and df.loc[i,'Availability']=='Yes':
                    SessInfo['bookingid']="UBER"+str(random_with_N_digits(6))
                    SessInfo['drivername']=random.choice(driver_name)
                    SessInfo['driverno']=random_with_N_digits(10)
                    SessInfo['carno']="KN-"+str(random_with_N_digits(2))+"-"+str(random_with_N_digits(4))
                    SessInfo['fare']=(df.loc[i,'Minimum Price']+df.loc[i,'Maximum Price'])/2
                    SessInfo['cabtype']=cabtype
                    #SessInfo['noofseats']=noofseats
                    SessInfo['eta']=df.loc[i,'ETA(min)'] 
                    SessInfo['distance']=df.loc[i,'Distance(Km)']
                    break


# # Function For Looking Cab #

# In[51]:

# when looking for a cab pass seatcount as 1
# when booking a cab pass the exact seat count required
def look_uber(source,destination=None):
    api = Geocoding()
    try:
        source_loc = api.geocode(source)
        source_lat_long = source_loc[0]['geometry']['location']
        start_lat=source_lat_long['lat']
        start_lng=source_lat_long['lng']
    except:
        print('Enter Valid Source.\n')
        print('Please let me know if you want to know anything more')
        return('null')
    
    
    if destination : 
        try:
            destination_loc = api.geocode(destination)
            destination_lat_long = destination_loc[0]['geometry']['location']
            end_lat=destination_lat_long['lat']
            end_lng=destination_lat_long['lng']

        except:
            print('Enter Valid destination.\n')
            print('Please let me know if you want to know anything more')
            return('null')
        
    else:
        end_lat=12.9173312 # latitude of Central Silk Board
        end_lng=77.6212483 # longitude of Central Silk Board
    try:
        response = client.get_price_estimates(
            start_latitude=start_lat,
            start_longitude=start_lng,
            end_latitude=end_lat,
            end_longitude=end_lng,
            seat_count=2
            )    

        estimate = response.json 
        cab_details_df= uber_cab_details(estimate)
        cab_details_df.fillna(0, axis=1, inplace=True)
        #print('Following are the cabs availability\n')
        #print(cab_details_df)
        return cab_details_df
    except uber_error.ClientError as e:
        print('Distance between two points exceeds 100 kms\n')
        print('Please let me know if you want to know anything more')
        return('null')
    


# # Storing Information #

# In[21]:

def storing_info(inp, intent, SessInfo):
    if intent == 'greetings':
        match1 = re.search('\\bhi|hello|hey\\b', inp)
        match2 = re.search('\\Good\sMorning|Good\sEvening|Good\sNight\\b', inp)
        SessInfo['greet1'] = -1
        SessInfo['greet2'] = -1
        if match1 : SessInfo['greet1'] = match1.group()
        if match2 : SessInfo['greet2'] = match2.group()    
        
    if intent == 'book' or intent == 'look':
    ### look for name entity and of 2 location are ther ethen look for from and 
    ### to and just store the one tagged with gpe
        source , destination = extract_location(inp)
        
        #print('am in storing')
        #print(source, destination)
        
        if source and destination:
            #print('both')
            SessInfo['source'] = source
            SessInfo['destination'] = destination
                        
        elif destination:
            #print('matched field is destination')
            '''check these updates'''
            SessInfo['destination'] = destination
            if 'source' in SessInfo: 
                #print('am here')
                del SessInfo['source']
        elif source:
            #print('matched field is source')
            SessInfo['source'] = source
            '''check these updates'''
            if 'destination' in SessInfo: del SessInfo['destination']
            
        cab_types = ['uberpool','ubergo','uberx', 'uberxl']
        for cab in cab_types:
            if cab in inp:
                SessInfo['cab_type_user'] = cab            


# # To Check If Source and Destination Field Is There Or Not #

# In[19]:

def handle_require_field(intent):
    ### keys needed for booking 
    if intent== 'book': requiredKey = ['source','destination']
    elif intent == 'look' : requiredKey = ['source']
    if SessInfo['almost_book'] == True : requiredKey.append('cab_type_user')
    
    ### storing missing key require to complete booking
    notAvail = []
    print(requiredKey)
    for key in requiredKey:
        if key not in SessInfo:
            notAvail.append(key)
    ### we can set flag and ask for all the keys simuntaneously
    if(len(notAvail)!=0):
        c = ['We need some details, please provide \n', 'Please provide the following details to proceed:\n', 'Can you give these details\n', 'We require more details, tell us about\n', 'Can you help us with some more details\n']
        print(random.choice(c))
        for key in notAvail:
            if key == 'cab_type_user':
                print(key)
                SessInfo[key] = input() 
                print('No of Seats')
                SessInfo['noofseats']=input()
            else:
                print(key)
                SessInfo[key] = input()


# # Confirming Source Destination #

# In[20]:

def confirm_src_destination():
    v = ['Please confirm your source {} {} is correct. Type\'Ok\'', 'Your details are {} {}. Type OK if it is correct', ' Your Source and Destination are {} {}. Type OK to confirm', ' Following are your details {} {}, Type OK to procced', 'Your Booking Source and Destinations are {} {}, Type OK to procced']
    print(random.choice(v).format(SessInfo.get('source',''),'and destination '+SessInfo['destination'] if 'destination' in SessInfo else '' ))
    confirm_inp = input()
    if confirm_inp.lower() == 'ok': 
        if 'destination' in SessInfo: SessInfo['almost_book']= True
    else:
        print('want to update source and destination. Reply with Yes')
        if input()=='yes':
            print('Sorry Provide the info again')
            if 'source' in SessInfo:
                print('source: ')
                SessInfo['source'] = input()
            if 'destination' in SessInfo:
                print('destination: ')
                SessInfo['destination'] = input()
            if 'destination' in SessInfo : SessInfo['almost_book']= True
        else:
            print('Lets have a fresh start')
            if 'destination' in SessInfo: del SessInfo['destination']
            if 'source' in SessInfo: del SessInfo['source']
            return


# # Function For Generating Response #

# In[245]:

def generating_response(inp, intent, SessInfo):
    greet=['Hey','Hi','Hello','Hey there','Hi Sir, how can i help you']
    if inp == 'exit':
        sys.exit()
    else:
        if intent == 'greetings':
            if SessInfo['greet1'] != -1:       
                print(random.choice(greet),'\n')
            elif SessInfo['greet2'] != -1:
                print(SessInfo['greet2'],'Sir \n Hope you have a nice day ahead \n')

        ### For handling response to booking queries
        
        if intent == 'look' :
            handle_require_field(intent)
            ### confirm and check if captured location and destination are coorect else update
            confirm_src_destination()
            
            ## call look function (source to destination#optional)
            x=['Looking for a cab from {} {}', ' Please wait! while we look for a cab from {} {}', 'We are lookin for a cab from {} {}', ' Wait a second , we are working on your cab booking from {} {}' ]
            y = ['Please Wait', 'Almost Done', 'Multiply 23*87 while we work on it', ' Add 323+23+456, lets see who is fast', 'Your Service is almost done', ' Count 5 4 3 2 1' ]
            print(random.choice(x).format(SessInfo.get('source',''),'to '+SessInfo['destination'] if 'destination' in SessInfo else '' ))
            print(random.choice(y)+'\n')
            response_avail_df = look_uber(SessInfo.get('source',''),SessInfo.get('destination',None))             
            if response_avail_df != 'null':
                z= ['Following are your results\n', 'Cabs for you\n', 'We find something for you\n', ' Have a look at the following options\n', ' Suitable cabs for you\n' ]
                print(random.choice(z))
                print(response_avail_df)
                print('Little friendly tip, You can for sure let us know your cab type preference and we will book it for you or enter any other query')
            return
        
        elif intent == 'book' :
            if SessInfo['almost_book'] == True:
                handle_require_field('book')
                #### book function           
            else:
                handle_require_field(intent)
                confirm_src_destination()
                response_avail_df = look_uber(SessInfo.get('source',''),SessInfo.get('destination',None))
                if response_avail_df != 'null':
                    print(response_avail_df)
                    handle_require_field('book')
                    available_cab_category = response_avail_df[response_avail_df['Availability']=='Yes']['Car type'].str.lower().tolist()
                    if SessInfo['cab_type_user'] in available_cab_category:
                ###book the cab
                        book(SessInfo['source'], SessInfo['destination'], SessInfo['cab_type_user'], SessInfo['noofseats'],response_avail_df,SessInfo)
                        print('Booking Done. Your booking id is {}'.format(SessInfo['bookingid']))
                        print('The estimated time of your cab\'s arrival is {}'.format(SessInfo['eta']))
                        print('Please let me know if you want to know anything more')
                #### if booking is done store flag that avoid user from asking to multiple book
                ### if asked explicitly to do second booking nullify the first one
                        return
                else:
                    print('Sorry {} not available. Please provide correct response with booking confirmation, if you want to book \n'.format(SessInfo['cab_type_user']))
                    return
            
            
        elif intent == 'duration':
            x1=['Looking for estimated time of arrival \n',' Please wait! while we look for Estimated Time of arrival of your cab \n','We are looking for estimated time of arrival\n' ]
            if 'bookingid' in list(SessInfo.keys()):
                print(random.choice(x1))
                print('The estimated time of arrival of your cab '+str(SessInfo['cabtype'])+' with booking id '+str(SessInfo['bookingid']) +' is '+str(SessInfo['eta'])+'\n')
            else:
                print("Sorry Sir, you haven't booked any cabs as of now")
                print("Please book a cab inorder to know the expected time of arrival")
                print('Please let me know if you want to know anything more')
                
                
        elif intent == 'fare_estimation':
                x2=['Looking for estimated fare \n',' Please wait! while we look for Estimated Fare \n','We are looking for estimated fare\n' ]
                z2= ['Following are your results\n', 'We find something for you\n', ' Have a look at the following options\n']
                if ('estimated' in inp) and ('bookingid' in list(SessInfo.keys())):
                    print(random.choice(x2))
                    print(random.choice(z2))
                    print('The estimated fare of your cab '+str(SessInfo['cabtype'])+' with booking id '+str(SessInfo['bookingid']) +' is '+str(SessInfo['fare'])+'\n')
                elif (('estimated' in inp) and ('from' in inp) and ('to' in inp)) or (('from' in inp) and ('to' in inp)):
                    src, dest=extract_location(inp)
                    df = look_uber(src, dest)
                    if df != 'null':
                        print(random.choice(x2))
                        print(random.choice(z2))
                        print("Following are the available cabs and their Minimum and Maximum price")
                        print(df[['Car type','Minimum Price','Maximum Price']])
                else:
                    print("Sorry Sir, you haven't booked any cabs as of now")
                    print("Please book a cab or give proper source and destination inorder to know the Minimum and Maximum price")
                    print('Please let me know if you want to know anything more')        
                    
                    
        elif intent == 'cancel':
            print('Are you sure that you want to cancel the cab Yes/No')
            decision=input().lower()
            if decision == 'yes':
                print('Your cab '+str(SessInfo['cabtype'])+' with booking id '+str(SessInfo['bookingid']) +' is cancelled')
                SessInfo=dict()          
            elif decision == 'no':
                print('Your cab is not cancelled')
                print('Please let me know if you want to know anything more')
                
        elif intent == 'schedule':
            
            
            
            
        else:
            print('I am Confused')


# # BOT Function #

# In[ ]:

### storing important info

def respond(self):
    SessInfo = dict() ## refresh dictionary when user work is completed
    SessInfo['almost_book'] = False
    #already_looked = False
    while(True):
        inp = input().lower()
        
        ### find the intent
        intent = find_intent(inp,nb_naive,vect)
        
        
        ### look for a particular patterns as per intent
        ### update the dict
        storing_info(inp, intent, SessInfo)

        ### generate output
        #### function1 (intent, dict) ## intent is greeting reply normally
        generating_response(inp, intent, SessInfo)

    for key in SessInfo:
        print(key, SessInfo[key])

