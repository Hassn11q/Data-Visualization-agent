# Data-Visualization-Agent
 a web-based application designed to provide interactive visualizations powered by Plotly 

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Usage](#example-usage)


## Overview

The **Data-Visualization-Agent** application enables users to:
- Upload CSV, Excel files to view data in a dynamic grid.
- Request customized visualizations through natural language inputs, which are transformed into Plotly visualizations by an integrated LLM model.
- Generate detailed data summaries to aid in visualization creation.

## Key Features

- **Data File Upload**: Supports CSV, Excel.
- **Data Grid Display**: Interactive grid for exploring uploaded data.
- **Natural Language Visualizations**: Input visualization requests in plain English.
- **LLM-Enhanced Plotting**: Uses Groq’s LLM to create responsive, insightful Plotly visualizations.
- **Data Summaries**: Generates summaries of numeric, categorical, and datetime data columns.
- **Error Handling**: Provides detailed error messages for unsupported formats and incorrect visualizations.

## Installation

### Prerequisites
- Python 3.9 or higher
- A valid `GROQ_API_KEY` for accessing Groq’s LLM

1. Clone this repository:
   ```bash
   git clone https://github.com/Hassn11q/Data-Visualization-agent.git
   cd Data-Visualization-agent
2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Create a .env file in the root directory to store your Groq API key:
   ```bash
   GROQ_API_KEY=your_groq_api_key

## Usage
1. Start the Application 
   ```bash 
   python app.py
2. Access the app at http://127.0.0.1:8050 in your web browser.
3. **Upload Data**:
- Click “Drag and Drop or Select a File” to upload your data file.
- Supported formats: CSV, Excel.
4. **Create Visualization**:
- Enter a request describing the visualization you want (e.g., “scatter plot of Sales vs. Profit”).
- Click “Generate Visualization” to produce a Plotly chart based on the request.
## Example Usage 
Below are some examples of natural language requests you can input to generate visualizations:

- "Create a line chart of temperature over time."
- "Visualize the distribution of product sales."
- "Scatter plot showing correlation between income and spending."
 
 The model will automatically analyze the data, select appropriate columns, and provide an optimal visualization.