from fast_dash import FastDash, Chat, Fastify, html, dmc, dbc
from fast_dash.utils import _pil_to_b64
from dash import Output, Input

import plotly.graph_objects as go
from flask import session

from langchain_core.messages import AIMessage, HumanMessage
from .agent import agent_executor
from .app_utils import Chatify, ABOUT_MARKDOWN

from dotenv import load_dotenv
load_dotenv()

Chat_md = Fastify(
    html.Div(
        style={
            "overflow-y": "scroll",
            "overflow-x": "hidden",
            "display": "flex",
            "flex-direction": "column-reverse",
        }
    ),
    "children"
)


suggested_questions = ["Show a comparison of the descriptors of benzene, toluene and cyclohexane along with their SMILES strings as a table",
                        "Generate a 2D visualization of toluene",
                        "SMILES string representation of a benzene molecule",
                        "How similar are benzene, toluene, and cyclohexane to each other based on RDKit fingerprints?",
                        "Is this material hazardous? URL: https://www.airgas.com/msds/001029.pdf"]

suggested_question_component = dmc.SegmentedControl(
                            data=suggested_questions,
                            value="",
                            orientation="vertical",
                            style={"background-color": "white", "flex-wrap": "wrap", "max-width": "100%", "white-space": "normal"},
                            fullWidth=False,
                            styles={
                                "label": {
                                    "text-align": "left",
                                    "flex-wrap": "wrap",
                                    "max-width": "100%",
                                    "word-wrap": "break-word", 
                                    "white-space": "normal"
                                }
                            },
                        )


def materials_agent(prompt: str,
                    suggestions: suggested_question_component) -> Chat_md:
    
    if prompt == "":
        agent = dict(query=prompt, response="Let me know what else I can do for you.")

    else:

        # Get chat history from Flask session
        chat_history = session.get("chat_history", [])

        chat_history_langchain = []
        for history in chat_history:
            chat_history_langchain.append(HumanMessage(content=history['query']))
            chat_history_langchain.append(AIMessage(content=history['response']))

        output = agent_executor.invoke({"input": prompt, "chat_history": chat_history_langchain})

        # Get tool outputs
        is_plotly = False
        tools = [o[0].tool for o in output['intermediate_steps']]
        tool_outputs = [o[1] for o in output['intermediate_steps']]

        answer = output.get("output", "Sorry! I'm busy right now.")
        cache_answer = answer
        for tool_name, tool_output in zip(tools, tool_outputs):
            if tool_name == "create_2D_visualization_from_SMILES":
                is_plotly = False
                answer = answer.split('![')[0]
                cache_answer = answer
                answer = f"{answer}\n\n![Molecule]({_pil_to_b64(tool_output)})"

            elif isinstance(tool_output, go.Figure):
                is_plotly = True

        agent = dict(query=prompt, response=answer)

        if is_plotly == True:
            agent.update(dict(plotly_figure=tool_output))

        # Record the conversation in the chat history
        chat_history.append(dict(query=prompt, response=cache_answer))
        session["chat_history"] = chat_history[-5:]

        agent = Chatify(agent)

    return agent


# Define and modify the app
app = FastDash(materials_agent, 
               port=8080,
               scale_height=1.5,
               update_live=False,
               theme="flatly",
               about=ABOUT_MARKDOWN,
               title_image_path="https://storage.googleapis.com/fast_dash/0.2.9/llm-materials-logo.png")

# Add a callback for suggested questions
@app.app.callback(Output("prompt", "value"),
                 Input("suggestions", "value"))
def update_input(s):
    return s

server = app.server
server.config["SECRET_KEY"] = "Super secret key"

if __name__ == "__main__":
    app.run()