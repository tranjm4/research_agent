# Research Assistant Agent

This project aims to create a chatbot agent that is able to answer user queries to retrieve, analyze, and organize academic sources to aid with research.

***

## Current Progress
- Refactoring [agent](https://github.com/tranjm4/research_agent/tree/main/src/backend/agent) to [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro) (MCP) 
- Refactoring [data pipeline](https://github.com/tranjm4/research_agent/tree/main/src/backend/data) to allow for more modular experimentation with data gathering, reducing data size (due to computation and storage limitations)

- Using new data from refactored pipeline to design new vectorstores to evaluate new models.

***

## Todo:
- Deploy Docker containers to AWS ECS
- Implement document management and annotation via agent
- Clean up user interface

***

## Challenges

- **PDF parsing bottleneck**: Computation for PDF parsing takes a long time due to the amount of text involved. For this reason, the project currently uses a limited amount of data for prototyping purposes.
    - Future work: Find cloud computing solutions to be able to efficiently parse PDFs
- **PDF parsing inaccuracies**: Current parsing methods sometimes creates weird artifacts in the resulting text, conjoining multiple adjacent words together. This is especially so for documents including mathematical notation.
    - Future work: Find more robust methods to parse PDFs, or dispatch certain parsing methods for documents involving math notation.
- **Keyword extraction inaccuracies**: Because of upstream PDF parsing failures, keyword extraction propagates the errors, as it often extracts the same weird artifacts produced by the PDF parsing step.
    - Addressing the PDF parsing first is necessary for revealing any isolated problems with keyword extraction.

***

## Primary Components

The project consists of:
- [data pipeline](https://github.com/tranjm4/research_agent/tree/main/src/backend/data)
- [agent server](https://github.com/tranjm4/research_agent/tree/main/src/backend/agent)
- [application backend server](https://github.com/tranjm4/research_agent/tree/main/src/backend/server)
- [application frontend client](https://github.com/tranjm4/research_agent/tree/main/src/interface)

### 1 Data Pipeline `src/data`

This is an **ETL pipeline** that harvests, transforms, versions, and stores data into a MongoDB instance for model experiments. Its primary components:

1. **harvester** ([`src/data/harvester/harvester.py`](https://github.com/tranjm4/research_agent/tree/main/src/backend/data/harvester))
2. **PDF parser** ([`src/data/pdf_parser/parser.py`](https://github.com/tranjm4/research_agent/tree/main/src/backend/data/pdf_parser))
3. **keyword extractor** ([`src/data/processing/extract_keywords.py`](https://github.com/tranjm4/research_agent/tree/main/src/backend/data/processing))
4. **text chunker** ([`src/data/processing/chunking.py`](https://github.com/tranjm4/research_agent/tree/main/src/backend/data/processing))

At each step, it checkpoints the stored data to a MongoDB instance (see [`src/data/mongo`](https://github.com/tranjm4/research_agent/tree/main/src/backend/data/mongo))

In between each step, these components read/write from/to a [**Kafka** message broker](https://github.com/tranjm4/research_agent/blob/main/src/backend/data/docker-compose.yaml) to allow for continuous data ingestion and processing as needed.


#### 1.1 Harvester `src/backend/data/harvester/harvester.py`

The harvester retrieves document metadata (upload date, document ID, abstract text, authors, primary topics, etc.) from **arXiv** using their [Open Archives Initiative (OAI)](https://info.arxiv.org/help/oa/index.html) API.

The API does not provide the PDF, so the PDF parser (next step) retrieves it.

#### 1.2 PDF parser `src/backend/data/pdf_parser/parser.py`

The parser uses PyMuPDF and PyMuPDF4LLM to parse documents into a markdown format, which is likely to produce more effective results for RAG purposes.

Given a document's ID parser makes a request to **arXiv** for the PDF, with which the parsing method of choice extracts the markdown text from it.

#### 1.3 Keyword extractor `src/backend/data/processing/extrac_keywords.py`

The keyword extractor is done for the entire document (to be shared across chunks for keyword search in reranking). This is done before chunking as opposed to the other way around due to computational costs; extracting from a single larger document would be much quicker than extracting from 100+ chunks.

#### 1.4 Text Chunker `src/backend/data/processing/chunking.py`

The text chunker currently utilizes various methods (for experimentation):
- **chunking by tokens**
    - using a tokenizer similarly used in ChatGPT, we partition the text into roughly equal-sized chunks by token length

- **chunking by sentences**
    - we preserve the sentences when chunking by tokens -- we don't split chunks mid-sentence to preserve overall ideas within a sentence

- **chunking by semantics**
    - we preserve the paragraphs when chunking by tokens -- paragraphs preserve similarly related ideas/semantics

### 2. Agent Server `src/backend/agent`

The agent server serves as an API endpoint to the **Application Backend Server**, and it is an MCP client to MCP servers that serve as tool and resource endpoints for agent tool/resource discovery.

Some key functionalities:
- Internal database search via the vectorstore of collected documents
- External search via DuckDuckGo API
- Document management features (e.g., annotation) (**TODO**)

### 3. Application Backend Server `src/backend/server`

The backend server serves multiple functions:
- Proxy to the application database and agent server
- Manage PostgreSQL database operations for user authentication, session management
- Manage PostgreSQL database operations for conversation management, document management
- Relay user prompts from frontend client to agent server and relay stream back to frontend client


### 4. Application Frontend Client `src/interface`

This serves as the end user's interface, developed with React.

Users should be able to
- Create an account
- Create multiple conversations with the agent
- Automate managing documents based on their conversations.
