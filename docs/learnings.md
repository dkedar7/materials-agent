## Learnings building Materials Agent

#### What would you tell a colleague who wants to get started using large language models for their research?
Advances in machine learning of large models have brought language models closer to mimicking human decision-making. As a result, LLMs have become adept at automating complex tasks, and developers do not require a thorough understanding of machine learning or language modeling.

The easiest way to get started is to brainstorm potential research applications having a templated operating procedure, which can be documented as a sequence of steps. In other words, LLMs can accurately mimic mundane and repetitive actions. Documenting such an operating procedure should be the next step if it doesn’t exist. Using closed-source LLM interfaces, like ChatGPT or OpenAI API playground, is the best way to evaluate the feasibility of using LLMs for the task.

Python is a great programming language for building software tools for further automation. Langchain and LlamaIndex are the most popular libraries for LLM operations and don't have a steep learning curve.

#### What did you learn during the hackathon? What surprised you about the usage of large language models?
Agent-based LLM workflows are becoming the norm for all LLM operations. They are very powerful because they can leverage LLM calls with a human-like thought process explicitly programmed in the prompts.

Their ability to understand Python function docstrings and discover the correct function arguments from a user query unlocks an unparalleled potential to integrate decades of work in building scientific research tools with human-like reasoning. Surprisingly, our “Materials Agent” can accurately determine the best tool (Python function) for a task from ten distinct tools. More work into realizing agents’ tool limits in the domain of materials science and chemistry will be of special interest.

#### What was easier or more challenging than expected? For the challenging aspects, did your team find creative solutions?
Langchain makes building LLM agents with access to tools convenient. Agents usually return data in the form of text. However, working with image outputs or outputs in non-standard data formats, like custom Python objects, is especially challenging. As part of Materials Agent, we used temporary data storage locations to save intermediate outputs before displaying them on the user interface. We believe that using inherited classes in Python with custom object representations and overwriting the default __repr__ method will be a promising approach.

Another common challenge with LLM operation libraries like Langchain is the frequency with which some methods and properties deprecate. Staying updated with the latest announcements, revisions, and procedures goes a long way.

The availability of open-source web development frameworks in Python made building a web application for Materials Agent easier than expected.

#### Which technique did you use? e.g., Retrieval Augmented Generation (RAG), Agents, Fine Tuning, Prompt Engineering.
Materials Agent’s biggest strength is its tool-calling (formerly known as function-calling) ability. We built the agent using Langchin, which facilitates using Pydantic objects and docstrings to easily represent Python tool outputs.

Prompt engineering was critical to getting the Materials Agent to return the expected responses. Prompt engineering played a huge role in two areas: one, where the agent was instructed to find the right tool for the job and summarize its response, and second, with the Python function (tool) docstrings.

Retrieval Augmented Generation (RAG) is one of the tools Materials Agent can access. When the user specifies a URL, the tool embeds it in a local SQLite database using Chroma, retrieves documents relevant to the user query, and summarizes a response using OpenAI’s GPT-3.5-turbo.

#### Which tools did you use? Langchain? Llama-index?
Langchain is our LLM operations library. We also used Plotly Dash and Fast Dash for web application development and the Plotly and Dash Bio molecule viewer to display molecule structures and other charts.

#### Which Large Language Model did you use? e.g., OpenAI, Llama, Claude
OpenAI’s GPT-3.5-Turbo across all tools.

#### Did you use your LLM to access existing data or generate new data?
There are three levels to Materials Agent’s data needs:
- **Existing data**: Most Materials Agent’s tools access existing data. These tools either fetch data from existing chemistry databases, like PDB files from the RCSB database, or generate figures from existing data.
- **LLM’s training data**: We observed that GPT-3.5-Turbo has some chemistry knowledge we need as part of its training data. We rely on it for some applications like retrieving molecule SMILES strings.
- **Generate new data**: Finally, tools like RAG access existing data and generate new data as summarized responses.
  
#### Is your data structured or unstructured?
Most of the data we deal with is unstructured text, but we also process structured PDB molecule structure files.

#### Why is an LLM particularly suited for your application?
Decades-long advances in computational chemistry and materials science have led to the development of widely used software tools. However, given the distinct foundations of these tools and steep learning curves, even computational chemists find it challenging to leverage them seamlessly. The situation is worse for scientists who don’t have any formal computational training. This reduces the adoption and accessibility of these tools immensely.

Materials Agent, which currently supports ten tools and lays down the framework to incorporate many more, aims to alleviate this adoption and accessibility challenge. It allows users to leverage the right tools without the need to write code or perform complex computations manually.

Some common computational chemistry workflows can be easily templated and documented in the form of operating procedures, inviting limitless opportunities to use LLMs.