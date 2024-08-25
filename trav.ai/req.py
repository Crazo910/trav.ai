import streamlit as st
import json
import requests

inputs = {"input_text": "I want to go to the museum"}
stream=requests.post(url="http://127.0.0.1:8000/query",data=json.dumps(inputs),timeout=600)                                     #crew.kickoff(inputs=inputs)
print(stream.json())