import React from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { useRepoStore } from './store/repoStore';

function App() {
  const { activeRepoId } = useRepoStore();

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans">
      <Sidebar />
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {activeRepoId ? (
          <ChatArea />
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500 flex-col gap-4">
            <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center">
              <span className="text-2xl">👋</span>
            </div>
            <p>Select a repository to start chatting</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
