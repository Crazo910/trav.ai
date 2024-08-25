import json
import streamlit as st
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union
from fastapi import FastAPI , Depends , HTTPException,Path
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain.schema import AgentFinish
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
import time 

import pandas as pd
import agentops
agentops.init("bbab0d9b-20e8-4353-b39c-418608576a7b")

df = pd.read_csv('madrid_landmarks.csv', dtype={
    'Landmark_ID': 'int64',
    'Name': 'object',
    'Type': 'object',
    'Address': 'object',
    'Latitude': 'float64',
    'Longitude': 'float64'
})


connection = sqlite3.connect("landmarks.db")
#df.to_sql(name="landmarks", con=connection)


@dataclass
class Event:
    event: str
    timestamp: str
    text: str


def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path: Path):
        self.log_path = log_path

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        assert len(prompts) == 1
        event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        generation = response.generations[-1][-1].message.content
        event = Event(event="llm_end", timestamp=_current_time(), text=generation)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event)) + "\n")



llm = ChatGroq(
    api_key="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",
    model_name="llama-3.1-70b-versatile",
    #callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))],
)


db = SQLDatabase.from_uri("sqlite:///landmarks.db")



@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")



@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)



@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)




@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})



sql_dev = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory=dedent(
        """
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `check_sql` to check your queries for correctness.
        Use the `execute_sql` to execute queries against the database.
    """
    ),
    llm=llm,
    tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=False,
)



data_analyst = Agent(
    role="Senior Data Analyst",
    goal="You receive data from the database developer and analyze it",
    backstory=dedent(
        """
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand and to the point. You have attention
        to detail and always produce very detailed work (as long as you need).

        The data you are going to be working with is landmark related. 
        The analysis you give is going to be used for event planning,
    """
    ),
    llm=llm,
    allow_delegation=False,
)


report_writer = Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing sightseeing event planning based on the landmark data recieved . """,
        backstory="""Specialist in travel planning and logistics with 
        decades of experience. You will pick the best starting landmark to vist based on location with end landmark that is fitting for the traveler """,
        
       
        llm=llm,
       # max_iter=10,
       # max_execution_time=3
       )



extract_data = Task(
    description="Extract data that is required for the query {query}.",
    expected_output="Database result for the query",
    agent=sql_dev,
)



analyze_data = Task(
    description="Analyze the data from the database and write an analysis for {query}.",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)




write_report = Task(
    description=dedent(
        """
         You will schedule out and pick the best starting landmark to vist based on location with end landmark that is fitting for the traveler .
           Also to your best knowledge give a fun fact about each landmark also. All the context will be given to you by your senior data analyst.Dont try to access thew data table.
    """
    ),
    expected_output="Markdown report",
    agent=report_writer,
    context=[analyze_data],
)





crew = Crew(
    agents=[sql_dev, data_analyst, report_writer],
    tasks=[extract_data, analyze_data, write_report],
    process=Process.sequential,
    verbose=2,
    memory=False,
    output_log_file="crew.log",
    #max_rpm=4,
    # embedder={
    #         "provider": "google",
    #         "config":{
    #             "model": 'models/embedding-001',
    #             "task_type": "retrieval_document",
    #             "title": "Embeddings for Embedchain"
    #         }
    # }
    
)



# inputs = {
#     "query": "I want to visit all the musuems"
# }

# result = crew.kickoff(inputs=inputs)
# print(result)


# import streamlit as st
# st.title("trav.ai")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# if prompt := st.chat_input("What would you like trav.ai to help you with?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     inputs = {
#     "query": prompt}
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         stream=crew.kickoff(inputs=inputs)
#         response = st.markdown(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    """Defines input for querying via FastAPI app"""

    input_text: str


class QueryOut(BaseModel):
    """Defines input for querying via FastAPI app"""

    
    output_text: str


@app.post("/query")
async def chat(
    query_in: QueryIn
) :
    # get output
    #st=time.time()
    # generate output_text
    input_text= {"query": query_in.input_text}       
    output_text = crew.kickoff(input_text)
    # et=time.time()
    # elapse_time=et-st
    # data = {"input_text": input_text, "output_text": output_text}
    # df=pd.DataFrame(data)
    # df.to_sql("chat_logs", con=connection, if_exists="append", index=True)
    
    # SQL_Q=text("SELECT * FROM chat_logs ORDER BY index DESC LIMIT 1;")
    # with connection.connect() as conn:
    #     result = conn.execute(SQL_Q)
    #     print(result)
    #     #conn.commit()
    
    # Results=result
    # table=pd.DataFrame(Results)
    

    return output_text











