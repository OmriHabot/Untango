import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Message, ToolCall } from '../store/chatStore';
import { User, Bot, Terminal, ChevronDown, ChevronRight, Check, X } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import clsx from 'clsx';

interface Props {
  message: Message;
}

const ToolCallDisplay: React.FC<{ toolCall: ToolCall }> = ({ toolCall }) => {
  const [isOpen, setIsOpen] = useState(false);
  const isCompleted = toolCall.status === 'completed';
  const isFailed = toolCall.status === 'failed';

  return (
    <div className="mb-2 rounded-md border border-slate-700 bg-slate-900 overflow-hidden text-sm">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 transition-colors text-left"
      >
        {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
        <Terminal className="w-4 h-4 text-purple-400" />
        <span className="font-mono text-slate-300 flex-1 truncate">
          {toolCall.tool}({Object.keys(toolCall.args).join(', ')})
        </span>
        {isCompleted && <Check className="w-4 h-4 text-green-500" />}
        {isFailed && <X className="w-4 h-4 text-red-500" />}
        {toolCall.status === 'running' && <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />}
      </button>
      
      {isOpen && (
        <div className="p-3 font-mono text-xs overflow-x-auto">
          <div className="mb-2">
            <div className="text-slate-500 mb-1">Arguments:</div>
            <pre className="text-slate-300 whitespace-pre-wrap">
              {JSON.stringify(toolCall.args, null, 2)}
            </pre>
          </div>
          {toolCall.result && (
            <div>
              <div className="text-slate-500 mb-1">Result:</div>
              <pre className="text-green-300/80 whitespace-pre-wrap max-h-60 overflow-y-auto">
                {toolCall.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const MessageBubble: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={clsx("flex gap-4 mb-6", isUser ? "flex-row-reverse" : "flex-row")}>
      <div className={clsx(
        "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
        isUser ? "bg-blue-600" : "bg-purple-600"
      )}>
        {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
      </div>

      <div className={clsx("flex-1 max-w-3xl min-w-0", isUser ? "items-end flex flex-col" : "")}>
        <div className={clsx(
          "rounded-lg p-4 text-slate-200",
          isUser ? "bg-blue-600/20 border border-blue-600/30" : "bg-slate-800/50 border border-slate-700"
        )}>
          {/* Tool Calls */}
          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mb-4 space-y-2">
              {message.toolCalls.map((tool, idx) => (
                <ToolCallDisplay key={idx} toolCall={tool} />
              ))}
            </div>
          )}

          {/* Message Content */}
          <div className="prose prose-invert prose-sm max-w-none break-words">
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <div className="rounded-md overflow-hidden my-2 border border-slate-700">
                      <div className="bg-slate-900 px-4 py-1 text-xs text-slate-400 border-b border-slate-700 flex justify-between items-center">
                        <span>{match[1]}</span>
                      </div>
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        customStyle={{ margin: 0, borderRadius: 0, background: '#0f172a' }}
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    </div>
                  ) : (
                    <code className="bg-slate-800 px-1 py-0.5 rounded text-purple-300 font-mono text-sm" {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          
          {/* Usage Stats */}
          {message.usage && (
            <div className="mt-2 text-xs text-slate-500 flex gap-3 border-t border-slate-700/50 pt-2">
              <span>Input: {message.usage.input_tokens}</span>
              <span>Output: {message.usage.output_tokens}</span>
              <span>Total: {message.usage.total_tokens}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
