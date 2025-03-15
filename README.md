# Knowledge Base Explorer

A simple Streamlit application that allows you to:
1. Upload PDF documents to build your knowledge base
2. Chat with an AI about the content of your documents
3. Explore suggested topics from your knowledge base

## Features

- **Document Upload**: Upload PDFs to build your knowledge base
- **Topic Exploration**: Get AI-suggested topics from your documents
- **Interactive Chat**: Ask questions about your documents
- **Persistent Storage**: All documents are stored in Supabase vector store

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Supabase Setup

1. Create a new Supabase project
2. Enable Vector store by following [Supabase Vector Store Setup](https://supabase.com/docs/guides/ai-vector-store)
3. Create a table named `documents` with the following SQL:
   ```sql
   create table documents (
     id bigserial primary key,
     content text,
     metadata jsonb,
     embedding vector(1536)
   );
   ```
4. Create a stored procedure named `match_documents` for similarity search:
   ```sql
   create or replace function match_documents (
     query_embedding vector(1536),
     match_count int DEFAULT 3
   ) returns table (
     id bigint,
     content text,
     metadata jsonb,
     similarity float
   )
   language plpgsql
   as $$
   begin
     return query
     select
       id,
       content,
       metadata,
       1 - (documents.embedding <=> query_embedding) as similarity
     from documents
     order by documents.embedding <=> query_embedding
     limit match_count;
   end;
   $$; 