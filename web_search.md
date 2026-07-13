# Tavily Web Search & Extract Tools Documentation

## Overview

This directory contains CLI tools for interacting with the Tavily API:

1. **Tavily Search** - Web search with AI-generated answers
2. **Tavily Extract** - Extract content from specific URLs

**API Key:** `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`

---

## Installation & Setup

The tool requires Node.js and axios. Dependencies are managed via `package.json`:

```bash
cd /home/interns/Desktop/work/parth/tavily
npm install
```

### SSL Certificate Issues (Windows/Corporate Networks)

If you encounter SSL certificate errors like `unable to verify the first certificate`, follow these steps:

#### For npm install:
```bash
npm config set strict-ssl false
npm install
```

#### For running the tool:

**Option 1: Set environment variable permanently (recommended)**
```bash
# Windows
setx NODE_OPTIONS "--use-system-ca"

# Then restart your terminal/VS Code
```

**Option 2: Set environment variable for current session**
```bash
# Windows CMD
set NODE_OPTIONS=--use-system-ca && node tavily-search.js -q "your query"

# PowerShell
$env:NODE_OPTIONS="--use-system-ca"; node tavily-search.js -q "your query"

# Linux/Mac
export NODE_OPTIONS="--use-system-ca"
node tavily-search.js -q "your query"
```

---

## Usage

### Help
```bash
node tavily-search.js -h
# or
node tavily-search.js --help
```

### Single Search Query
```bash
node tavily-search.js -q "your search query"
```

**Example:**
```bash
node tavily-search.js -q "What is Tavily AI?"
```

### Advanced Search with Options
```bash
# Advanced depth with more results
node tavily-search.js -q "quantum computing" -d advanced -n 10

# News from last 7 days
node tavily-search.js -q "electric vehicles" -t news --days 7

# Science topic with JSON output
node tavily-search.js -q "CRISPR gene editing" -t science -o crispr-results.json
```

### Batch Search
Create a JSON file with an array of queries:

**example-queries.json:**
```json
[
  "What is Tavily AI?",
  "Latest developments in quantum computing 2026",
  "Electric vehicle market trends"
]
```

Run batch search:
```bash
node tavily-search.js -f example-queries.json -o results.json
```

---

## Command Line Options

| Flag | Alias | Description | Default |
|------|-------|-------------|---------|
| `--query` | `-q` | Single search query text | - |
| `--file` | `-f` | Path to JSON file with array of queries | - |
| `--depth` | `-d` | Search depth: `"basic"` or `"advanced"` | `basic` |
| `--topic` | `-t` | Topic category: `"general"`, `"news"`, or `"science"` | `general` |
| `--days` | - | Number of days back for results (1-30) | - |
| `--max` | `-n` | Maximum results per query (1-20) | `5` |
| `--output` | `-o` | Save results to JSON file | - |
| `--help` | `-h` | Show help message | - |

---

## Topic Types

| Topic | Description |
|-------|-------------|
| `general` | General web search across all topics |
| `news` | News articles and recent events |
| `science` | Scientific papers, research, and academic content |

---

## Search Depth

| Depth | Description |
|-------|-------------|
| `basic` | Quick search, faster results, good for simple queries |
| `advanced` | Comprehensive search, more thorough, better for complex topics |

---

## Output Format

### Console Output
The tool displays formatted results in the terminal:
- Search query header
- AI-generated answer (if available)
- List of results with:
  - Title
  - URL
  - Content snippet
  - Relevance score

### JSON Output
When using `-o`, results are saved as JSON:
```json
[
  {
    "query": "What is Tavily AI?",
    "success": true,
    "answer": "Tavily is a web intelligence platform...",
    "results": [
      {
        "title": "...",
        "url": "...",
        "content": "...",
        "score": 0.91
      }
    ],
    "response_time": "..."
  }
]
```

---

## NPM Scripts

The `package.json` includes convenient scripts:

```json
{
  "scripts": {
    "search": "node tavily-search.js",
    "batch": "node tavily-search.js -f example-queries.json -o results.json"
  }
}
```

**Usage:**
```bash
npm run search -- -q "your query"
npm run batch
```

---

## Example Commands

```bash
# Basic search
node tavily-search.js -q "latest AI developments"

# Advanced search with 10 results
node tavily-search.js -q "quantum computing breakthroughs" -d advanced -n 10

# News from last 3 days
node tavily-search.js -q "tech layoffs" -t news --days 3

# Science search
node tavily-search.js -q "mRNA vaccine research" -t science

# Batch search with output
node tavily-search.js -f queries.json -o batch-results.json

# Combined options
node tavily-search.js -q "renewable energy" -t news -d advanced --days 14 -n 15 -o energy-report.json
```

---

## Files Structure

```
tavily/
├── tavily-search.js      # Main CLI script
├── package.json          # Node.js dependencies and scripts
├── example-queries.json  # Example batch queries
├── results.json          # Sample output file
└── web_search.md         # This documentation
```

---

## Error Handling

The tool handles errors gracefully:
- Invalid API responses display error messages
- Missing files show "File not found" errors
- Network errors are caught and reported
- Summary shows successful vs failed queries

---

## Notes for AI Assistant

When the user asks you to search for something:

1. **Use the CLI tool** by executing: `node tavily-search.js -q "<query>"`
2. **Choose appropriate options** based on the query type:
   - News queries → use `-t news --days 7`
   - Scientific topics → use `-t science`
   - Complex topics → use `-d advanced`
   - Need more results → use `-n 10` or higher
3. **Save results** if the user might want them later → use `-o filename.json`
4. **For multiple topics** → create a queries JSON file and use batch mode

**Example response pattern:**
```
Let me search for that information using Tavily.

[Execute: node tavily-search.js -q "query" -n 5]

Based on the search results, here's what I found...
```

---

## Tavily Extract CLI Tool

### Overview
The Extract tool allows you to extract content from specific URLs. It returns the raw text content from web pages.

### Help
```bash
node tavily-extract.js -h
# or
node tavily-extract.js --help
```

### Single URL Extraction
```bash
node tavily-extract.js -u "https://example.com"
```

**Example:**
```bash
node tavily-extract.js -u "https://en.wikipedia.org/wiki/Artificial_intelligence"
```

### Extract with Character Limit
```bash
node tavily-extract.js -u "https://example.com/article" -l 10000
```

### Batch Extraction
Create a JSON file with an array of URLs:

**example-urls.json:**
```json
[
  "https://en.wikipedia.org/wiki/Artificial_intelligence",
  "https://example.com"
]
```

Run batch extraction:
```bash
node tavily-extract.js -f example-urls.json -o extract-results.json
```

### Quiet Mode (JSON only)
```bash
node tavily-extract.js -u "https://example.com" -q
```

---

## Extract Command Line Options

| Flag | Alias | Description | Default |
|------|-------|-------------|---------|
| `--url` | `-u` | Single URL to extract content from | - |
| `--file` | `-f` | Path to JSON file with array of URLs | - |
| `--limit` | `-l` | Maximum characters to extract | `5000` |
| `--output` | `-o` | Save results to JSON file | - |
| `--quiet` | `-q` | Only output JSON, no formatted output | `false` |
| `--help` | `-h` | Show help message | - |

---

## Extract Output Format

### Console Output
The tool displays formatted results:
- URL header
- Extracted content (truncated to 500 chars for display)
- Favicon URL (if available)
- Images count (if any)
- Response time
- Credits used

### JSON Output
When using `-o` or `-q`, results are saved/output as JSON:
```json
[
  {
    "urls": "https://example.com",
    "success": true,
    "results": [
      {
        "url": "https://example.com",
        "title": "Example Domain",
        "raw_content": "Full page content...",
        "images": []
      }
    ],
    "failed_results": [],
    "response_time": 0.01,
    "request_id": "..."
  }
]
```

---

## Files Structure (Updated)

```
tavily/
├── tavily-search.js      # Search CLI script
├── tavily-extract.js     # Extract CLI script
├── package.json          # Node.js dependencies and scripts
├── example-queries.json  # Example search queries
├── example-urls.json     # Example URLs for extraction
├── results.json          # Sample search output
├── extract-results.json  # Sample extract output
└── web_search.md         # This documentation
```

---

## API Reference

### Tavily Search Endpoint
```
POST https://api.tavily.com/search
```

### Request Body
```json
{
  "api_key": "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",
  "query": "search query",
  "search_depth": "basic",
  "topic": "general",
  "days": 7,
  "max_results": 5,
  "include_answer": true,
  "include_raw_content": false
}
```

### Response
```json
{
  "answer": "AI-generated answer",
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com",
      "content": "Snippet content",
      "score": 0.95
    }
  ],
  "response_time": "0.5s"
}
```

### Tavily Extract Endpoint
```
POST https://api.tavily.com/extract
```

### Request Headers
```
Authorization: Bearer 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
Content-Type: application/json
```

### Request Body
```json
{
  "urls": ["https://example.com"],
  "limit_chars": 5000
}
```

### Response
```json
{
  "results": [
    {
      "url": "https://example.com",
      "title": "Page Title",
      "raw_content": "Full extracted text content...",
      "images": []
    }
  ],
  "failed_results": [],
  "response_time": 0.01,
  "usage": {
    "credits": 1
  },
  "request_id": "..."
}
```
