from crewai import Crew
from textwrap import dedent
from agents import TripAgents
from tasks import TripTasks
from langchain.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai import Agent, Crew, Process, Task

load_dotenv()

class TripCrew:

  def __init__(self, origin):
    #self.cities = cities
     self.origin = origin
    # self.interests = interests
    # self.date_range = date_range

  def run(self):
    agents = TripAgents()
    tasks = TripTasks()

    # city_selector_agent = agents.city_selection_agent()
    # local_expert_agent = agents.local_expert()
    # travel_concierge_agent = agents.travel_concierge()
    DB_developer_agent=agents.sql_dev()
    Data_analyst=agents.data_analyst()
    Writer_agent=agents.report_writer()
  
    extract_data_task=tasks.extract_data(
      DB_developer_agent
    )

    analyze_task=tasks.analyze_data(
      Data_analyst,extract_data_task
    )

    report_writer_task=tasks.report_writer(
      Writer_agent,analyze_task)
    

    # identify_task = tasks.identify_task(
    #   city_selector_agent,
    #   self.origin,
    #   self.cities,
    #   self.interests,
    #   self.date_range
    # )
    # gather_task = tasks.gather_task(
    #   local_expert_agent,
    #   self.origin,
    #   self.interests,
    #   self.date_range
    # )
   

    crew = Crew(
      agents=[
        DB_developer_agent, Data_analyst, Writer_agent
      ],
      tasks=[extract_data_task, analyze_task, report_writer_task],
      process=Process.sequential,
    verbose=2
    )
    inputs={"query":self.origin}
    result = crew.kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
  print("## Welcome to Trip Planner Crew")
  print('-------------------------------')
  location = input(
    dedent("""
      What landmark activities are we interested in 
    """))
  # cities = input(
  #   dedent("""
  #     What are the cities options you are interested in visiting?
  #   """))
  # date_range = input(
  #   dedent("""
  #     What is the date range you are interested in traveling?
  #   """))
  #  interests = input(
  #   dedent("""
  #      What are some of your high level interests and hobbies?
  #    """))
  
  trip_crew = TripCrew(location)
  result = trip_crew.run()
  print("\n\n########################")
  print("## Here is you Trip Plan")
  print("########################\n")
  print(result)