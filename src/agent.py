import os
import json
import requests
import litellm
from dotenv import load_dotenv

load_dotenv()

def web_search_tool(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is missing."
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3
    }
    
    try:
        response = requests.post(url, json=payload)
        results = response.json().get("results", [])
        snippets = [f"Source: {r['url']}\nContent: {r['content']}" for r in results]
        return "\n\n".join(snippets) if snippets else "No relevant results found."
    except Exception as e:
        return f"Search failed due to an error: {str(e)}"
    
def run_agentic_search(model_name: str, user_question: str, system_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    search_count = 0
    max_turns = 5
    search_steps = []

    for turn in range(max_turns):
        response = litellm.completion(
            model=model_name,
            messages=messages,
            tools=tools_spec,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message.model_dump())

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "web_search_tool":
                    search_count += 1
                    args = json.loads(tool_call.function.arguments)
                    search_results = web_search_tool(args["query"])
                    search_steps.append({
                        "iteration": search_count,
                        "model_thought": response_message.content or "",
                        "tavily_query": args["query"],
                        "tavily_results": search_results
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "web_search_tool",
                        "content": search_results
                    })
            continue
        else:
            return {
                "answer": response_message.content,
                "search_queries_count": search_count,
                "total_tokens": response.get("usage", {}).get("total_tokens", 0),
                "search_steps": search_steps
            }

    return {"answer": "Timeout: Max turns reached without final answer.", "search_queries_count": search_count, "total_tokens": 0, "search_steps": search_steps}

def run_prefetch_search(model_name: str, user_question: str, system_prompt: str, api_base: str = None, api_key: str = None) -> dict:
    tavily_results = web_search_tool(user_question)

    user_content = (
        f"Here are the latest search results:\n\n"
        f"{tavily_results}\n\n"
        f"Based ONLY on the search results above, answer this question:\n{user_question}"
    )

    extra = {}
    if api_base:
        extra["api_base"] = api_base
    if api_key:
        extra["api_key"] = api_key

    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        **extra,
    )

    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "search_queries_count": 1,
        "total_tokens": response.get("usage", {}).get("total_tokens", 0),
        "search_steps": [{
            "iteration": 1,
            "model_thought": "",
            "tavily_query": user_question,
            "tavily_results": tavily_results,
        }],
    }