/**
 * @file apiService.ts
 * @description API client for the FastAPI backend. Replaces direct Gemini API calls.
 * Provides functions to interact with the backend RAG endpoints.
 */

import { Message, HandbookDoc, ToolStep } from '../types';

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
/**
 * Streaming RAG query using Server-Sent Events.
 * Calls onToolStep for each tool invocation as it happens,
 * then returns the final response.
 */
export async function getHandbookResponseStreamed(
  query: string,
  history: Message[],
  chatId: string | undefined,
  onToolStep: (step: ToolStep) => void,
) {
  try {
    const response = await fetch(`${API_URL}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        history: history.map(m => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp
        })),
        chat_id: chatId ?? null
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResult: any = null;

    // Queue tool steps and reveal them with a staggered delay
    const STEP_DELAY_MS = 600;
    const stepQueue: ToolStep[] = [];
    let stepOrder = 0;

    const flushStepQueue = async () => {
      for (const step of stepQueue) {
        onToolStep(step);
        await new Promise(r => setTimeout(r, STEP_DELAY_MS));
      }
      stepQueue.length = 0;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (eventType === 'tool_step') {
            const step = data as ToolStep;
            // Expand plan_searches into individual steps for staggered display
            if (step.tool_name === 'plan_searches' && Array.isArray(step.arguments?.queries)) {
              for (const q of step.arguments.queries as string[]) {
                stepOrder++;
                stepQueue.push({ tool_name: 'search_handbook', arguments: { query: q }, order: stepOrder });
              }
            } else {
              stepOrder++;
              stepQueue.push({ ...step, order: stepOrder });
            }
            await flushStepQueue();
          } else if (eventType === 'done') {
            finalResult = data;
          }
        }
      }
    }

    return finalResult || {
      content: "I'm having trouble connecting right now. Please try again in a moment.",
      sources: [],
      tool_steps: [],
      isError: true
    };
  } catch (error) {
    console.error("Streaming API Error:", error);
    return {
      content: "I'm having trouble connecting to the knowledge base right now. Please try again in a moment.",
      sources: [],
      tool_steps: [],
      chat_id: null,
      interaction_id: null,
      isError: true
    };
  }
}

/**
 * Non-streaming RAG query (kept as fallback).
 */
export async function getHandbookResponse(query: string, history: Message[], chatId?: string) {
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
        })),
        chat_id: chatId ?? null
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
      sources: [],
      chat_id: null,
      interaction_id: null,
      isError: true
    };
  }
}
