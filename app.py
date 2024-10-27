import os
from typing import Optional, Tuple, Dict, Any
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import base64
import io
from datetime import datetime
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize LLM
groq_llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-70b-versatile")

def get_fig_from_code(code: str, df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Safely execute the generated code to create a Plotly figure.
    
    Args:
        code (str): The Python code to execute
        df (pd.DataFrame): The DataFrame to use in the code
        
    Returns:
        Optional[go.Figure]: The generated Plotly figure or None if there's an error
    """
    try:
        # Create a safe namespace with necessary imports
        namespace = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'make_subplots': make_subplots,
            'df': df
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Return the figure if it exists
        if 'fig' in namespace and isinstance(namespace['fig'], (go.Figure, dict)):
            return namespace['fig']
        else:
            raise ValueError("Code did not create a valid Plotly figure")
            
    except Exception as e:
        print(f"Error executing code: {str(e)}")
        raise

def parse_contents(contents: str, filename: str) -> Tuple[Optional[Dict], Optional[str], html.Div]:
    """
    Parse uploaded file contents into a DataFrame.
    
    Args:
        contents (str): The contents of the uploaded file
        filename (str): The name of the uploaded file
        
    Returns:
        Tuple[Optional[Dict], Optional[str], html.Div]: The parsed data, filename, and grid component
    """
    try:
        # Decode content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Read file into DataFrame
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' in filename.lower() or 'tsv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
        else:
            return None, None, html.Div(['Unsupported file type. Please upload a CSV, Excel, or TSV file.'])

        # Convert datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        # Create grid component
        grid = html.Div([
            html.H5(filename),
            dag.AgGrid(
                rowData=df.to_dict('records'),
                columnDefs=[{'field': i, 'sortable': True, 'filter': True} for i in df.columns],
                defaultColDef={
                    'resizable': True,
                    'sortable': True,
                    'filter': True,
                    'floatingFilter': True
                },
                dashGridOptions={'pagination': True, 'paginationAutoPageSize': True}
            ),
            html.Hr()
        ])
        
        return df.to_dict('records'), filename, grid
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None, html.Div([f'Error processing file: {str(e)}'])

def generate_data_summary(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to summarize
        
    Returns:
        str: A detailed summary of the DataFrame
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    summary = []
    
    # Basic info
    summary.append(f"Total Rows: {len(df)}")
    summary.append(f"Total Columns: {len(df.columns)}")
    
    # Column types
    if len(numeric_cols) > 0:
        summary.append(f"\nNumeric Columns: {', '.join(numeric_cols)}")
        # Add statistics for numeric columns
        stats_df = df[numeric_cols].describe()
        summary.append("\nNumeric Statistics:")
        summary.append(stats_df.to_string())
    
    if len(categorical_cols) > 0:
        summary.append(f"\nCategorical Columns: {', '.join(categorical_cols)}")
        # Add value counts for categorical columns
        summary.append("\nCategory Distributions:")
        for col in categorical_cols:
            summary.append(f"\n{col}:")
            summary.append(df[col].value_counts().head().to_string())
    
    if len(datetime_cols) > 0:
        summary.append(f"\nDateTime Columns: {', '.join(datetime_cols)}")
        # Add date ranges
        summary.append("\nDate Ranges:")
        for col in datetime_cols:
            summary.append(f"{col}: {df[col].min()} to {df[col].max()}")
    
    return "\n".join(summary)

# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For production deployment

# App layout
app.layout = html.Div([
    # Storage
    dcc.Store(id='stored-data', storage_type='memory'),
    dcc.Store(id='stored-file-name', storage_type='memory'),
    
    # Header
    html.H1("Interactive Data Visualization AI", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # File Upload Section
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File', style={'color': '#3498db'})
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'backgroundColor': '#f8f9fa'
            },
            multiple=False
        )
    ]),
    
    # Data Grid
    html.Div(id='grid-container'),
    
    # Visualization Request Section
    html.Div([
        dcc.Textarea(
            id='user-request',
            placeholder='Describe the visualization you want to create...',
            style={
                'width': '100%',
                'height': 100,
                'margin': '20px 0',
                'padding': '10px',
                'borderRadius': '5px',
                'borderColor': '#bdc3c7'
            }
        ),
        html.Button(
            'Generate Visualization',
            id='submit-request',
            style={
                'backgroundColor': '#2ecc71',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'marginBottom': '20px'
            }
        )
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    # Results Section
    dcc.Loading(
        children=[
            html.Div(id='my-figure', style={'margin': '20px 0'}),
            dcc.Markdown(id='content', style={'margin': '20px 0', 'padding': '10px',
                                            'backgroundColor': '#f8f9fa',
                                            'borderRadius': '5px'})
        ],
        type='cube',
        color='#2ecc71'
    )
])

@callback(
    Output("stored-data", "data"),
    Output("stored-file-name", "data"),
    Output("grid-container", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def update_output(contents: Optional[str], filename: Optional[str]) -> Tuple[Optional[Dict], Optional[str], html.Div]:
    """
    Handle file upload and update the data store.
    """
    if contents is None:
        return None, None, ''
    
    return parse_contents(contents, filename)

@callback(
    Output("my-figure", "children"),
    Output("content", "children"),
    Input("submit-request", "n_clicks"),
    State("user-request", "value"),
    State("stored-data", "data"),
    State("stored-file-name", "data"),
    prevent_initial_call=True
)
def create_graph(n_clicks: Optional[int], 
                user_input: Optional[str], 
                file_data: Optional[Dict], 
                file_name: Optional[str]) -> Tuple[Any, str]:
    """
    Generate visualization based on user request and data.
    """
    if not n_clicks or not user_input or not file_data:
        return "", "Please upload a file and enter a visualization request."

    try:
        # Convert data to DataFrame
        df = pd.DataFrame(file_data)
        
        # Generate data summary
        data_summary = generate_data_summary(df)
        
        # Create prompt
        enhanced_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert data visualization specialist using Plotly. Your task is to create the most effective and visually appealing visualization based on the following data and user request.

DATA SUMMARY:
{data_summary}

USER REQUEST:
{user_input}

Create a professional visualization following these requirements:

1. DATA ANALYSIS
- Use appropriate data types and transformations
- Handle missing values if necessary
- Scale or normalize data if needed
- Consider statistical relationships

2. VISUALIZATION BEST PRACTICES
- Choose appropriate chart type
- Use clear labels and titles
- Implement proper color schemes
- Add informative tooltips
- Include legends where needed
- Set appropriate axes ranges
- Use optimal layout settings

3. CODE REQUIREMENTS
- Use the 'df' DataFrame
- Create a Plotly figure named 'fig'
- Include necessary imports
- Implement error handling
- DO NOT include fig.show()

4. ENHANCED FEATURES
- Interactive elements
- Proper formatting
- Responsive layout
- Professional styling
- Clear annotations

Provide ONLY the Python code that creates the visualization."""),
            HumanMessage(content="Create the visualization following the above specifications.")
        ])

        # Generate visualization code
        chain = enhanced_prompt | groq_llm
        response = chain.invoke({})
        
        # Extract code
        code_match = re.search(r'```(?:python)?(.*?)```', response.content, re.DOTALL)
        if not code_match:
            return "", "No valid Python code found in the response. Please try again."
        
        # Clean and execute code
        code = code_match.group(1).strip()
        fig = get_fig_from_code(code, df)
        
        return dcc.Graph(figure=fig), "Visualization generated successfully!"

    except Exception as e:
        return "", f"Error creating visualization: {str(e)}\n\nPlease try a different request."

if __name__ == '__main__':
    app.run_server(debug=True)