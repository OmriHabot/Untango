import { create } from 'zustand';
import { api, ChatMessage } from '../api/client';

export interface ToolCall {
  tool: string;
  args: Record<string, any>;
  result?: string;
  status: 'running' | 'completed' | 'failed';
}

export interface Message extends ChatMessage {
  id: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  isLoading: boolean; // Added
  currentToolCall: ToolCall | null; // Renamed from currentTool
  
  addMessage: (message: Message) => void;
  updateLastMessage: (content: string) => void;
  sendMessage: (content: string, model?: string) => Promise<void>; // Kept model optional for implementation
  clearMessages: () => Promise<void>; // Renamed from clearChat, now async
  loadHistory: () => Promise<void>; // Added
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  isLoading: false, // Added
  currentToolCall: null, // Renamed from currentTool

  loadHistory: async () => {
    set({ isLoading: true });
    try {
      const { history } = await api.getChatHistory();
      // Map backend history to frontend message format
      // Note: Backend history doesn't store tool calls yet, just text
      const formattedMessages: Message[] = history.map((msg: any) => ({
        role: msg.role,
        content: msg.content,
        id: Math.random().toString(36).substring(7), // Generate temp ID
        timestamp: Date.now()
      }));
      set({ messages: formattedMessages });
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  clearMessages: async () => { // Renamed from clearChat
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
      timestamp: Date.now()
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
      toolCalls: []
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
              updateLastMessage(fullContent);
            } else if (event.type === 'tool_start') {
              set((state) => {
                const msgs = [...state.messages];
                const last = msgs[msgs.length - 1];
                // Create a new array for toolCalls to ensure immutability
                const newToolCalls = [...(last.toolCalls || [])];
                
                const newTool: ToolCall = {
                  tool: event.tool,
                  args: event.args,
                  status: 'running'
                };
                newToolCalls.push(newTool);
                
                // Update the message with new toolCalls
                const newLast = { ...last, toolCalls: newToolCalls };
                msgs[msgs.length - 1] = newLast;

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
                    // Find the running tool call matching this tool
                    // We search from the end to find the most recent one
                    let toolIndex = -1;
                    for (let i = newToolCalls.length - 1; i >= 0; i--) {
                        if (newToolCalls[i].tool === event.tool && newToolCalls[i].status === 'running') {
                            toolIndex = i;
                            break;
                        }
                    }
                    
                    if (toolIndex !== -1) {
                        newToolCalls[toolIndex] = {
                            ...newToolCalls[toolIndex],
                            result: event.result,
                            status: 'completed'
                        };
                        
                        const newLast = { ...last, toolCalls: newToolCalls };
                        msgs[msgs.length - 1] = newLast;
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
              updateLastMessage(fullContent);
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
