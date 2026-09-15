import os
import operator
from langchain_core.prompts import prompt
import psycopg
import json
from pathlib import Path

from typing import TypedDict, Annotated,List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from app.services.flight_service import extract_flight_info
from langchain_core.messages import(
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_groq import ChatGroq

from app.tools.tavily_tool import tavily_search
from app.tools.flight_tool import flight_search

from config import settings
from app.utils.airport_helper import get_airport_code
from app.schemas.travel_schema import TravelDetails
from app.services.travel_parser import extract_travel_details

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    api_key=settings.GROQ_API_KEY,
)

#db
DATABASE_URL = settings.DATABASE_URL
# State
class TravelState(TypedDict):
    messages: Annotated[List[AnyMessage],operator.add]
    user_query: str
    travel_details: TravelDetails
    flight_result: str
    hotel_result: str
    itinerary: str
    llm_calls: int

# Parser_agent
def parser_agent(state: TravelState):

    details = extract_travel_details(
        state["user_query"]
    )

    return {
        "travel_details": details,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# flight agent
def flight_agent(state: TravelState):
    details = state["travel_details"]
    # query = state['user_query']
    # flight_info = extract_flight_info(
    #         query
    #     )

    dep_iata = get_airport_code(
            details.departure_city
        )

    arr_iata = get_airport_code(
            details.destination_city
        )
    flight_data = flight_search(details,
        dep_iata,
        arr_iata,
    )
    print("\n===== Flight Tool Output =====")
    print(flight_data)
    return{
        "flight_result": flight_data,
        "messages":[
            AIMessage(content = f'Flight result fetched')
        ],
        "llm_calls": state.get('llm_calls',0) +1
    }

# hostel agent
# hold
def hotel_agent(state: TravelState):
    details = state["travel_details"]
    user_query = state["user_query"]

    search_query = f"""
    Find hotels in {details.destination_city}.

    Budget: {details.budget if details.budget else "Not specified"}

    User requirements:
    {user_query}
    """

    hotel_data = tavily_search(search_query)

    return {
        "hotel_result": hotel_data,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
# itinerary agent
def itinerary_agent(state: TravelState):
    prompt = f"""
    Travel Details:
    {json.dumps(state["travel_details"].model_dump(), indent=2)}
    # {state["travel_details"]}
    
    Flights:
    {json.dumps(state["flight_result"], indent=2)}

    Hotels:
    {json.dumps(state["hotel_result"], indent=2)}

    Create a personalized travel itinerary based on the information.
    Generate:
    1. A day-wise travel itinerary.
    2. Recommended sightseeing places.
    3. Local transportation suggestions.
    4. Food recommendations.
    5. Estimated travel tips.

"""

    response = llm.invoke([
        SystemMessage(
            content='You are an expert travel planner who creates detailed, practical, and personalized travel itineraries'
        ),
        HumanMessage(content=prompt),
   ])

    return{
    'planner':response.content,
    'messages':[response],
    'llm_calls':state.get('llm_calls',0)+1,

   }

#final response agent
def final_agent(state:TravelState):
    final_prompt = f"""
    Generate the final travel plan for the user.

    Flight Details:
    {state["flight_result"]}

    Hotel Details:
    {state["hotel_result"]}

    Travel Itinerary:
    {state["itinerary"]}

    Present the response in a clean and user-friendly format.
"""

    response = llm.invoke([
        SystemMessage(content = 'You are an expert travel assistant.'),
        HumanMessage(content =final_prompt)
    ])

    return {
        'messages':[response],
        'llm_calls': state.get('llm_calls',0) + 1
    }

# build graph
graph = StateGraph(TravelState)

graph.add_node('parser_agent',parser_agent)
graph.add_node('flight_agent',flight_agent)
graph.add_node('hotel_agent',hotel_agent)
graph.add_node('itinerary_agent',itinerary_agent)
graph.add_node('final_agent',final_agent)

graph.add_edge(START, 'parser_agent')
graph.add_edge('parser_agent', 'flight_agent')
graph.add_edge('flight_agent', 'hotel_agent')
graph.add_edge("hotel_agent", 'itinerary_agent')
graph.add_edge("itinerary_agent", 'final_agent')
graph.add_edge("final_agent", END)

# Database connection:
conn = psycopg.connect(DATABASE_URL)
checkpointer = PostgresSaver(conn=conn)
checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer,
)

if __name__== '__main__':
    config = {
        'configurable': {
            'thread_id': 'avi123',
        }
    }
    user_query = input("Enter you travel request:\n")

    result = app.invoke(
        {
            "messages":[HumanMessage(content=user_query)],
            "user_query": user_query,
            "flight_result": [],
            "hotel_result": [],
            "itinerary": "",
            "llm_calls": 0,

        },
        config = config
    )

# its optional
    outdir_dir = Path('output')
    outdir_dir.mkdir(exist_ok=True)

    file_path = outdir_dir / 'travel_plant.md'

    with open(file_path, "w", encoding="utf-8") as file:
        for i, msg in enumerate(result["messages"], start=1):
            file.write(f"# Agent {i}\n\n")
            file.write(msg.content)
            file.write("\n\n")
            file.write("-" * 80)
            file.write("\n\n")

    print(f"Saved to {file_path}")


    print("\n========== FINAL RESPONSE ==========\n")

    for msg in result['messages']:
        print(msg.content)
