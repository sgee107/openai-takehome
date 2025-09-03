import { NextRequest, NextResponse } from 'next/server';
import { ChatSearchResponse, SearchRequest } from '../../../lib/types';

// Backend API URL - use environment variable or default to localhost
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, topK = 20, lambda = 0.85, debug = false } = body;
    
    // Validate input
    if (!query || typeof query !== 'string' || !query.trim()) {
      return NextResponse.json(
        { error: 'Query is required and must be a non-empty string' },
        { status: 400 }
      );
    }

    // Prepare request for backend
    const searchRequest: SearchRequest = {
      query: query.trim(),
      topK,
      lambda,
      debug
    };

    console.log(`[Frontend] Forwarding search request to backend: ${JSON.stringify(searchRequest)}`);

    // Forward to backend vNext API
    const backendResponse = await fetch(`${BACKEND_URL}/chat/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(searchRequest),
      // Add timeout
      signal: AbortSignal.timeout(30000) // 30 second timeout
    });

    if (!backendResponse.ok) {
      console.error(`[Frontend] Backend API error: ${backendResponse.status} ${backendResponse.statusText}`);
      
      // Try to get error details from backend
      let errorMessage = 'Backend API error';
      try {
        const errorData = await backendResponse.json();
        errorMessage = errorData.detail || errorMessage;
      } catch (e) {
        // Ignore JSON parsing errors for error responses
      }
      
      return NextResponse.json(
        { error: errorMessage },
        { status: backendResponse.status }
      );
    }

    const data: ChatSearchResponse = await backendResponse.json();
    
    console.log(`[Frontend] Received ${data.results.length} results from backend`);
    if (debug) {
      console.log(`[Frontend] Debug info:`, data.debug);
    }

    return NextResponse.json(data);

  } catch (error: any) {
    console.error('[Frontend] API error:', error);
    
    // Handle specific error types
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      return NextResponse.json(
        { error: 'Request timeout - please try again' },
        { status: 504 }
      );
    }
    
    if (error.code === 'ECONNREFUSED') {
      return NextResponse.json(
        { error: 'Backend service unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to process search request' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ 
    message: 'Fashion Search API vNext - use POST to search',
    version: '1.0.0',
    endpoints: {
      search: 'POST /api/search'
    }
  });
}