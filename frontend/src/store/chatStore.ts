import { create } from 'zustand';
import { api, ChatMessage } from '../api/client';

export interface ToolCall {
  tool: string;
  args: Record<string, any>;
  result?: string;
  status: 'running' | 'completed' | 'failed';
}

export interface MessagePart {
  type: 'text' | 'tool';
  content?: string;
  toolCall?: ToolCall;
}

export interface Message extends ChatMessage {
  id: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  parts?: MessagePart[]; // Added for ordered rendering
  usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  isLoading: boolean;
  currentToolCall: ToolCall | null;
  
  addMessage: (message: Message) => void;
  updateLastMessage: (content: string) => void;
  sendMessage: (content: string, model?: string) => Promise<void>;
  clearMessages: () => Promise<void>;
  loadHistory: () => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  isLoading: false,
  currentToolCall: null,

  loadHistory: async () => {
    set({ isLoading: true });
    try {
      const { history } = await api.getChatHistory();
      // Map backend history to frontend message format
      const formattedMessages: Message[] = history.map((msg: any) => ({
        role: msg.role,
        content: msg.content,
        id: Math.random().toString(36).substring(7),
        timestamp: Date.now(),
        // For history, we might not have parts if it's old data, 
        // but we can try to reconstruct or just leave it as content-only.
        // If the backend sends parts in history later, we can use them.
        // For now, just content.
        parts: [{ type: 'text', content: msg.content }]
      }));
      set({ messages: formattedMessages });
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  clearMessages: async () => {
    set({ isLoading: true });
    try {
      await api.clearChatHistory();
      set({ messages: [], currentToolCall: null });
    } catch (error) {
      console.error('Failed to clear history:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  
  updateLastMessage: (content) => set((state) => {
    const newMessages = [...state.messages];
    if (newMessages.length > 0) {
      const lastMsg = newMessages[newMessages.length - 1];
      if (lastMsg.role === 'model') {
        lastMsg.content = content;
      }
    }
    return { messages: newMessages };
  }),

  clearChat: () => set({ messages: [] }),

  sendMessage: async (content: string, model = 'gemini-2.5-flash') => {
    const { addMessage, updateLastMessage } = get();
    
    // Add user message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: Date.now(),
      parts: [{ type: 'text', content }]
    };
    addMessage(userMsg);

    set({ isStreaming: true });

    // Add placeholder model message
    const modelMsgId = (Date.now() + 1).toString();
    const modelMsg: Message = {
      id: modelMsgId,
      role: 'model',
      content: '',
      timestamp: Date.now(),
      toolCalls: [],
      parts: []
    };
    addMessage(modelMsg);

    try {
      // Prepare history for API
      const history = get().messages.slice(0, -1).map(m => ({
        role: m.role,
        content: m.content
      }));

      const response = await fetch('http://localhost:8001/chat-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...history, { role: 'user', content }],
          model
        }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            
            if (event.type === 'token') {
              fullContent += event.content;
              
              set((state) => {
                const msgs = [...state.messages];
                const last = msgs[msgs.length - 1];
                const newParts = [...(last.parts || [])];
                
                // If last part is text, append to it
                if (newParts.length > 0 && newParts[newParts.length - 1].type === 'text') {
                  newParts[newParts.length - 1] = {
                    ...newParts[newParts.length - 1],
                    content: (newParts[newParts.length - 1].content || '') + event.content
                  };
                } else {
                  // Otherwise start new text part
                  newParts.push({ type: 'text', content: event.content });
                }
                
                msgs[msgs.length - 1] = { ...last, content: fullContent, parts: newParts };
                return { messages: msgs };
              });

            } else if (event.type === 'tool_start') {
              set((state) => {
                const msgs = [...state.messages];
                const last = msgs[msgs.length - 1];
                const newToolCalls = [...(last.toolCalls || [])];
                const newParts = [...(last.parts || [])];
                
                const newTool: ToolCall = {
                  tool: event.tool,
                  args: event.args,
                  status: 'running'
                };
                newToolCalls.push(newTool);
                newParts.push({ type: 'tool', toolCall: newTool });
                
                msgs[msgs.length - 1] = { ...last, toolCalls: newToolCalls, parts: newParts };

                return { 
                  messages: msgs, 
                  currentToolCall: newTool
                };
              });
            } else if (event.type === 'tool_end') {
              set((state) => {
                const msgs = [...state.messages];
                const last = msgs[msgs.length - 1];
                
                if (last.toolCalls) {
                    const newToolCalls = [...last.toolCalls];
                    const newParts = [...(last.parts || [])];

                    // Find the running tool call matching this tool
                    let toolIndex = -1;
                    for (let i = newToolCalls.length - 1; i >= 0; i--) {
                        if (newToolCalls[i].tool === event.tool && newToolCalls[i].status === 'running') {
                            toolIndex = i;
                            break;
                        }
                    }
                    
                    if (toolIndex !== -1) {
                        const updatedTool = {
                            ...newToolCalls[toolIndex],
                            result: event.result,
                            status: 'completed' as const
                        };
                        newToolCalls[toolIndex] = updatedTool;
                        
                        // Also update the tool in parts
                        // We need to find the part that corresponds to this tool call
                        // Since we push to both arrays in sync, we can try to find the part with the same tool name and running status
                        // Or simpler: just find the last tool part that matches
                        for (let i = newParts.length - 1; i >= 0; i--) {
                            if (newParts[i].type === 'tool' && 
                                newParts[i].toolCall?.tool === event.tool && 
                                newParts[i].toolCall?.status === 'running') {
                                newParts[i] = { type: 'tool', toolCall: updatedTool };
                                break;
                            }
                        }

                        msgs[msgs.length - 1] = { ...last, toolCalls: newToolCalls, parts: newParts };
                        return { messages: msgs, currentToolCall: null };
                    }
                }
                return {};
              });
            } else if (event.type === 'usage') {
              set((state) => {
                const msgs = [...state.messages];
                const last = msgs[msgs.length - 1];
                last.usage = event.usage;
                return { messages: msgs };
              });
            } else if (event.type === 'error') {
              console.error('Stream error:', event.content);
              fullContent += `\n\n*[Error: ${event.content}]*`;
              set((state) => {
                 const msgs = [...state.messages];
                 const last = msgs[msgs.length - 1];
                 const newParts = [...(last.parts || [])];
                 newParts.push({ type: 'text', content: `\n\n*[Error: ${event.content}]*` });
                 msgs[msgs.length - 1] = { ...last, content: fullContent, parts: newParts };
                 return { messages: msgs };
              });
            }
          } catch (e) {
            console.error('Error parsing stream line:', line, e);
          }
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      updateLastMessage(get().messages[get().messages.length - 1].content + '\n\n*[Failed to send message]*');
    } finally {
      set({ isStreaming: false, currentToolCall: null });
    }
  }
}));
