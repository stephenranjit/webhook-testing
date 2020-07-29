#!/usr/local/bin/python3


from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from google.auth.transport.requests import Request

# Define the required scopes
scopes = [
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/firebase.database"
]

# Authenticate a credential with the service account
credentials = service_account.Credentials.from_service_account_file(
    "/Users/sranjit/development/scripts/sranjit-herbalinfo-firebase-adminsdk-fl62i-8db7b763a3.json", scopes=scopes)

# Use the credentials object to authenticate a Requests session.
authed_session = AuthorizedSession(credentials)
response = authed_session.get(
    "https://sranjit-herbalinfo.firebaseio.com/users/ada/name.json")

# Or, use the token directly, as described in the "Authenticate with an
# access token" section below. (not recommended)
request = Request()
credentials.refresh(request)
access_token = credentials.token

#print(access_token)

'''
import ssl
hostname='oauth2.googleapis.com'
port=443

f = open('cert.der','wb')
cert = ssl.get_server_certificate((hostname, port))
f.write(ssl.PEM_cert_to_DER_cert(cert))



import requests
r = requests.get('https://oauth2.googleapis.com/token')

print(r.text)

'''