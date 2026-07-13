#!/usr/bin/env node

import axios from 'axios';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { argv } from 'process';
import https from 'https';

// Use system CA certificates for HTTPS requests
const httpsAgent = new https.Agent({
  rejectUnauthorized: false,
});

const TAVILY_API_KEY = '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';

const tavilyClient = axios.create({
  baseURL: 'https://api.tavily.com',
  httpsAgent,
  headers: {
    'Content-Type': 'application/json',
  },
});

function printHelp() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                    Tavily Search CLI Tool                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  node tavily-search.js [options]

Options:
  -q, --query <text>       Single search query
  -f, --file <path>        Batch search from JSON file with array of queries
  -d, --depth <level>      Search depth: "basic" or "advanced" (default: basic)
  -t, --topic <type>       Topic: "general", "news", or "science" (default: general)
  --days <num>             Days back for results (1-30, mainly for news)
  -n, --max <num>          Max results per query (1-20, default: 5)
  -o, --output <path>      Save results to JSON file
  -h, --help               Show this help message

Examples:
  # Single search
  node tavily-search.js -q "latest AI developments"

  # Advanced search with more results
  node tavily-search.js -q "quantum computing" -d advanced -n 10

  # News search from last 7 days
  node tavily-search.js -q "electric vehicles" -t news --days 7

  # Batch search from file
  node tavily-search.js -f queries.json -o results.json

  # Science search with output
  node tavily-search.js -q "CRISPR gene editing" -t science -o crispr-results.json
`);
}

async function performSearch(query, options) {
  const { searchDepth = 'basic', topic = 'general', days, maxResults = 5 } = options;

  try {
    const response = await tavilyClient.post('/search', {
      api_key: TAVILY_API_KEY,
      query,
      search_depth: searchDepth,
      topic,
      days,
      max_results: maxResults,
      include_answer: true,
      include_raw_content: false,
    });

    return { success: true, data: response.data, query };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return {
        success: false,
        error: error.response?.data?.detail ?? error.message,
        query,
      };
    }
    throw error;
  }
}

function formatResult(result, index) {
  let output = '';
  
  output += `\n${'═'.repeat(70)}\n`;
  output += `QUERY: ${result.query}\n`;
  output += `${'═'.repeat(70)}\n\n`;

  if (!result.success) {
    output += `❌ Error: ${result.error}\n`;
    return output;
  }

  const data = result.data;

  if (data.answer) {
    output += `📝 ANSWER:\n${data.answer}\n\n`;
  }

  if (data.results && data.results.length > 0) {
    output += `📊 RESULTS (${data.results.length} found):\n\n`;
    
    data.results.forEach((item, idx) => {
      output += `${idx + 1}. ${item.title}\n`;
      output += `   URL: ${item.url}\n`;
      output += `   📄 ${item.content}\n`;
      if (item.score) output += `   ⭐ Score: ${item.score}\n`;
      output += `\n`;
    });
  } else {
    output += 'No results found.\n';
  }

  return output;
}

function formatResultsJson(results) {
  return results.map(result => {
    if (!result.success) {
      return { query: result.query, success: false, error: result.error };
    }
    return {
      query: result.query,
      success: true,
      answer: result.data.answer,
      results: result.data.results,
      response_time: result.data.response_time,
    };
  });
}

async function main() {
  const args = argv.slice(2);

  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    printHelp();
    return;
  }

  const options = {
    query: null,
    file: null,
    searchDepth: 'basic',
    topic: 'general',
    days: undefined,
    maxResults: 5,
    output: null,
  };

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '-q':
      case '--query':
        options.query = next;
        i++;
        break;
      case '-f':
      case '--file':
        options.file = next;
        i++;
        break;
      case '-d':
      case '--depth':
        options.searchDepth = next;
        i++;
        break;
      case '-t':
      case '--topic':
        options.topic = next;
        i++;
        break;
      case '--days':
        options.days = parseInt(next, 10);
        i++;
        break;
      case '-n':
      case '--max':
        options.maxResults = parseInt(next, 10);
        i++;
        break;
      case '-o':
      case '--output':
        options.output = next;
        i++;
        break;
    }
  }

  const results = [];

  // Single query search
  if (options.query) {
    console.log(`\n🔍 Searching for: "${options.query}"...`);
    const result = await performSearch(options.query, options);
    results.push(result);
    console.log(formatResult(result));
  }

  // Batch search from file
  if (options.file) {
    if (!existsSync(options.file)) {
      console.error(`❌ File not found: ${options.file}`);
      return;
    }

    const queries = JSON.parse(readFileSync(options.file, 'utf-8'));
    
    if (!Array.isArray(queries)) {
      console.error('❌ File must contain a JSON array of query strings');
      return;
    }

    console.log(`\n📁 Batch searching ${queries.length} queries...\n`);

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      console.log(`[${i + 1}/${queries.length}] Searching: "${query}"...`);
      const result = await performSearch(query, options);
      results.push(result);
      console.log(formatResult(result));
      
      // Small delay between requests
      if (i < queries.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
  }

  // Save results to file
  if (options.output) {
    const jsonResults = formatResultsJson(results);
    writeFileSync(options.output, JSON.stringify(jsonResults, null, 2));
    console.log(`\n💾 Results saved to: ${options.output}`);
  }

  // Summary
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`✅ Summary: ${successful} successful, ${failed} failed`);
  console.log(`${'═'.repeat(70)}\n`);
}

main().catch(console.error);
