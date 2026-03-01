/**
 * @file apiService.ts
 * @description API client for the FastAPI backend. Replaces direct Gemini API calls.
 * Provides functions to interact with the backend RAG endpoints.
 */

import { Message, HandbookDoc } from '../types';

// Determine API URL based on environment:
// - If VITE_BACKEND_URL is explicitly set, use it (can be set to empty string for same-origin)
// - If in production and no VITE_BACKEND_URL, use empty string for relative URLs
// - Otherwise, default to localhost:9481 for local development
const API_URL = import.meta.env.VITE_BACKEND_URL !== undefined
  ? import.meta.env.VITE_BACKEND_URL
  : (import.meta.env.PROD ? '' : 'http://localhost:9481');

/**
 * Fetches all handbook documents from the backend.
 */
export async function getHandbookDocs(): Promise<HandbookDoc[]> {
  try {
    const response = await fetch(`${API_URL}/api/handbook`);
    if (!response.ok) {
      throw new Error(`Failed to fetch handbook: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching handbook documents:", error);
    return [];
  }
}

/**
 * Executes a RAG query to generate a handbook-backed response.
 * Replaces the direct Gemini API call from geminiService.ts
 */
export async function getHandbookResponse(query: string, history: Message[]) {
  try {
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        history: history.map(m => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp
        }))
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("API Communication Error:", error);
    return {
      content: "I'm having trouble connecting to the knowledge base right now. Please try again in a moment.",
      sources: []
    };
  }
}

/**
 * Triggers a secondary semantic analysis pass.
 * Finds specific phrases in the source document that support a previously generated answer.
 * Used for the "Highlight with AI" feature in the SourceViewer.
 */
export async function getRelevanceHighlights(answer: string, documentContent: string): Promise<string[]> {
  try {
    const response = await fetch(`${API_URL}/api/highlights`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        answer,
        document_content: documentContent
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.highlights;
  } catch (error) {
    console.error("Semantic Analysis Error:", error);
    return [];
  }
}
