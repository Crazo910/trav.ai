
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union

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

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

from tools.list_tables import list_tables
from tools.tables_schema import tables_schema
from tools.execute_sql import execute_sql
from tools.check_sql import check_sql




import pandas as pd


llm = ChatGroq(
    api_key="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",
    model_name="llama3-70b-8192",
    #callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))],
)

class TripAgents():
  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True,
        llm=llm)

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True,
        llm=llm)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        verbose=True,
        llm=llm)

  def report_writer(self):
   return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing sightseeing event planning based on the landmark data recieved . """,
        backstory="""Specialist in travel planning and logistics with 
        decades of experience. You will pick the best starting landmark to vist based on location with end landmark that is fitting for the traveler """,llm=llm,
    allow_delegation=False,)
  
  def data_analyst(self):
    return  Agent(
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
    allow_delegation=False,)
  
  def sql_dev(self):
    return Agent(
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
    #tools=[list_tables, tables_schema, execute_sql, check_sql],
    allow_delegation=False,)





