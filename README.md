# Second Brain Assistant

An AI-powered knowledge base assistant that helps users explore and interact with their personal knowledge base through engaging conversations and progressive disclosure.

## Features

- Interactive conversation with AI-driven topic suggestions
- User profile management with interests and projects
- Document processing (PDF, TXT, MD) with vector storage
- Progressive disclosure of information
- Engaging topic exploration with follow-up questions

## Prerequisites

- Python 3.11+
- OpenAI API key
- Supabase account and project

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/second-brain-assistant.git
cd second-brain-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

5. Set up Supabase:
- Create a new Supabase project
- Create the required tables using the SQL in `create_user_profiles.sql`
- Enable vector storage for your project
- Add your Supabase credentials to `.env`

## Development

Run the application locally:
```bash
streamlit run app.py
```

## Production Deployment

### Using Docker

1. Build the Docker image:
```bash
docker build -t second-brain-assistant .
```

2. Run the container:
```bash
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  --name second-brain-assistant \
  second-brain-assistant
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service role key
- `STREAMLIT_SERVER_PORT`: (Optional) Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: (Optional) Server address (default: 0.0.0.0)

## Maintenance

- Logs are stored in the `logs` directory
- Temporary files are stored in the `temp` directory
- Both directories are automatically created if they don't exist

## Security Considerations

- Never commit `.env` file or any sensitive credentials
- Use environment variables for all sensitive information
- Regularly update dependencies for security patches
- Monitor API usage and set appropriate rate limits

## Troubleshooting

Common issues and solutions:

1. OpenAI API Rate Limits:
   - Implement appropriate retry mechanisms
   - Monitor usage and adjust as needed

2. Supabase Connection Issues:
   - Check network connectivity
   - Verify credentials and permissions
   - Monitor database usage

3. Document Processing Errors:
   - Check file permissions
   - Verify supported file formats
   - Monitor temp directory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here] 