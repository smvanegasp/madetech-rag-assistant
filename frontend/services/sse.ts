/**
 * @file sse.ts
 * @description Incremental Server-Sent Events parser for fetch() streams (EventSource can't POST).
 *
 * Network chunks can split an event anywhere (CloudFront and Lambda streaming do), so the
 * partial line and the current event name carry over between chunks. Comment lines (": ping",
 * the backend's keep-alive) and unknown fields are ignored; a blank line ends an event.
 */

export interface SseEvent {
  event: string;
  data: string;
}

export function createSseParser() {
  let buffer = '';
  let eventType = '';
  let dataLines: string[] = [];

  /** Feed decoded text; returns the events completed by it, in order. */
  return function feed(text: string): SseEvent[] {
    const events: SseEvent[] = [];
    buffer += text;
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const raw of lines) {
      const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
      if (line === '') {
        if (dataLines.length > 0) {
          events.push({ event: eventType || 'message', data: dataLines.join('\n') });
        }
        eventType = '';
        dataLines = [];
      } else if (line.startsWith(':')) {
        continue;
      } else if (line.startsWith('event:')) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^ /, ''));
      }
    }
    return events;
  };
}
