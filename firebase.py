#!/usr/local/bin/python3

import requests
import json
import firebaseoauth


#print(firebaseoauth.access_token)

#payload = {
#            "title": 'Mrs'
#        }

class FirebaseDB():

	def __init__(self):
		pass

	def insert(self, payload):
		r = requests.put(
        "https://sranjit-herbalinfo.firebaseio.com/benefits/herbs/latin_names.json",
        headers = {
            'Authorization': 'Bearer ' + firebaseoauth.access_token,
            'Content-Type': 'application/json; UTF-8',
        },
    #headers={
    #    "Authorization": "access_token "+firebaseoauth.access_token
    #},
        data = json.dumps(payload)
        )
        #print(r.headers)
        #print(r.json())

	#def delete






'''
curl -X PUT -d '{ "first": "Jack", "last": "Sparrow" }' \
  'https://sranjit-herbalinfo.firebaseio.com/users/jack/name.json? \
  access_token=ya29.c.KpAB1Qch_bsh-y1q_AlWlZDlI7VoymmKjhnLvouvVzTOzCWF7fPXl3AXnjqSKup1ewn49YJveTmlXjhnBz40kN_zPoyuSgWm313MpnpKd0v-XRvxsQTqk5_rYtYsRcZfYFIIh3szWzsAcyJMdjNcdEkJl6S5XJni32inWXD_lBaWU1P0d-EB4oD28wg_1ZZrKmmV&print=silent'
'''