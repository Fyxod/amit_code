#!/usr/bin/env node

import axios from 'axios';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { argv } from 'process';
import https from 'https';

// Use system CA certificates for HTTPS requests
// Note: If you encounter SSL issues, run with: NODE_OPTIONS="--use-system-ca" node tavily-extract.js
const TAVILY_API_KEY = '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';

const tavilyClient = axios.create({
  baseURL: 'https://api.tavily.com',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${TAVILY_API_KEY}`,
  },
  httpsAgent: new https.Agent({
    rejectUnauthorized: false, // Set to true in production with proper CA setup
  }),
});

function printHelp() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                   Tavily Extract CLI Tool                        ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  node tavily-extract.js [options]

Options:
  -u, --url <url>          Single URL to extract content from
  -f, --file <path>        Batch extract from JSON file with array of URLs
  -l, --limit <chars>      Maximum characters to extract (default: 5000)
  -o, --output <path>      Save results to JSON file
  -q, --quiet              Only output JSON, no formatted console output
  -h, --help               Show this help message

Examples:
  # Single URL extraction
  node tavily-extract.js -u "https://en.wikipedia.org/wiki/Artificial_intelligence"

  # Extract with character limit
  node tavily-extract.js -u "https://example.com/article" -l 10000

  # Batch extraction from file
  node tavily-extract.js -f urls.json -o results.json

  # Quiet mode (JSON only)
  node tavily-extract.js -u "https://example.com" -q

  # Save results to file
  node tavily-extract.js -u "https://example.com" -o extract-results.json
`);
}

async function extractContent(urls, options) {
  const { limit = 5000 } = options;

  try {
    const response = await tavilyClient.post('/extract', {
      urls: Array.isArray(urls) ? urls : [urls],
      limit_chars: limit,
    });

    return { success: true, data: response.data, urls };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return {
        success: false,
        error: error.response?.data?.detail ?? error.message,
        urls,
      };
    }
    throw error;
  }
}

function formatResult(result) {
  let output = '';
  
  output += `\n${'═'.repeat(70)}\n`;
  
  const urls = Array.isArray(result.urls) ? result.urls : [result.urls];
  output += `URLS: ${urls.join(', ')}\n`;
  output += `${'═'.repeat(70)}\n\n`;

  if (!result.success) {
    output += `❌ Error: ${result.error}\n`;
    return output;
  }

  const data = result.data;

  if (data.results && data.results.length > 0) {
    output += `📄 EXTRACTED CONTENT (${data.results.length} pages):\n\n`;
    
    data.results.forEach((item, idx) => {
      output += `${idx + 1}. ${item.url}\n`;
      output += `   ─${'─'.repeat(68)}\n`;
      
      // Truncate content for display if too long
      const content = item.raw_content || '';
      const displayContent = content.length > 500 
        ? content.substring(0, 500) + '... [truncated]' 
        : content;
      output += `   ${displayContent.replace(/\n/g, '\n   ')}\n`;
      
      if (item.favicon) {
        output += `   🖼️  Favicon: ${item.favicon}\n`;
      }
      if (item.images && item.images.length > 0) {
        output += `   📷 Images: ${item.images.length} found\n`;
      }
      output += `\n`;
    });
  }

  if (data.failed_results && data.failed_results.length > 0) {
    output += `⚠️  FAILED (${data.failed_results.length}):\n`;
    data.failed_results.forEach((item, idx) => {
      output += `   - ${item.url}: ${item.error || 'Unknown error'}\n`;
    });
    output += `\n`;
  }

  if (data.response_time) {
    output += `⏱️  Response Time: ${data.response_time}s\n`;
  }
  if (data.usage && data.usage.credits) {
    output += `💳 Credits Used: ${data.usage.credits}\n`;
  }

  return output;
}

function formatResultsJson(results) {
  return results.map(result => {
    if (!result.success) {
      return { urls: result.urls, success: false, error: result.error };
    }
    return {
      urls: result.urls,
      success: true,
      results: result.data.results,
      failed_results: result.data.failed_results,
      response_time: result.data.response_time,
      usage: result.data.usage,
      request_id: result.data.request_id,
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
    url: null,
    file: null,
    limit: 5000,
    output: null,
    quiet: false,
  };

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case '-u':
      case '--url':
        options.url = next;
        i++;
        break;
      case '-f':
      case '--file':
        options.file = next;
        i++;
        break;
      case '-l':
      case '--limit':
        options.limit = parseInt(next, 10);
        i++;
        break;
      case '-o':
      case '--output':
        options.output = next;
        i++;
        break;
      case '-q':
      case '--quiet':
        options.quiet = true;
        break;
    }
  }

  const results = [];

  // Single URL extraction
  if (options.url) {
    if (!options.quiet) {
      console.log(`\n🔍 Extracting content from: "${options.url}"...`);
    }
    const result = await extractContent(options.url, options);
    results.push(result);
    if (!options.quiet) {
      console.log(formatResult(result));
    }
  }

  // Batch extraction from file
  if (options.file) {
    if (!existsSync(options.file)) {
      console.error(`❌ File not found: ${options.file}`);
      return;
    }

    const urls = JSON.parse(readFileSync(options.file, 'utf-8'));
    
    if (!Array.isArray(urls)) {
      console.error('❌ File must contain a JSON array of URL strings');
      return;
    }

    if (!options.quiet) {
      console.log(`\n📁 Batch extracting ${urls.length} URLs...\n`);
    }

    for (let i = 0; i < urls.length; i++) {
      const url = urls[i];
      if (!options.quiet) {
        console.log(`[${i + 1}/${urls.length}] Extracting: "${url}"...`);
      }
      const result = await extractContent(url, options);
      results.push(result);
      if (!options.quiet) {
        console.log(formatResult(result));
      }
      
      // Small delay between requests
      if (i < urls.length - 1) {
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
  
  if (!options.quiet) {
    console.log(`\n${'═'.repeat(70)}`);
    console.log(`✅ Summary: ${successful} successful, ${failed} failed`);
    console.log(`${'═'.repeat(70)}\n`);
  } else {
    // In quiet mode, output JSON to console
    console.log(JSON.stringify(formatResultsJson(results), null, 2));
  }
}

main().catch(console.error);
