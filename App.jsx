import React, { useState, useRef, useEffect } from "react";

// =====================================================================
// 🎨 Sleek SVG Icons (Premium aesthetics, no external assets needed)
// =====================================================================
const SendIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
  </svg>
);

const BotIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-emerald-400">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" />
  </svg>
);

const UserIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-slate-300">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
  </svg>
);

const CitationIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-emerald-400 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-.621-.504-1.125-1.125-1.125H9.75M3 16.061V4.419c0-.847.67-1.57 1.517-1.614 3.78-.2 7.564-.2 11.34 0 .848.044 1.518.767 1.518 1.614v11.642c0 .847-.67 1.57-1.517 1.614-3.78.2-7.564.2-11.34 0A1.516 1.516 0 013 16.061z" />
  </svg>
);

const DatabaseIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-emerald-500 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75m-16.5-3.75v3.75" />
  </svg>
);

export default function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! I am your Self-Correcting RAG assistant. Ask me any biomedical question, and I will search our local scientific database, self-correct any hallucinations using NLI, and provide verified, sentence-level citations.",
      citations: []
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [hoveredCitation, setHoveredCitation] = useState(null); // Tracks bidirectional hover triggers (chunk_id)
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Handle message sending to FastAPI
  const handleSend = async (e) => {
    e.preventDefault();
    const query = input.trim();
    if (!query) return;

    // Add user query to chat history
    setMessages((prev) => [...prev, { sender: "user", text: query }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query })
      });

      if (!response.ok) {
        throw new Error(`API returned server error: ${response.status}`);
      }

      const data = await response.json();
      
      // Support both output schemas defensively:
      // (answer vs draft_answer, and citations vs verified_citations)
      const answerText = data.answer || data.draft_answer || "";
      const citationsArray = data.citations || data.verified_citations || [];

      // Add bot response with citations
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: answerText,
          citations: citationsArray
        }
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Error: Unable to connect to your FastAPI backend. Please make sure api.py is running on http://localhost:8000.\n\nDetails: ${error.message}`,
          citations: []
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // =====================================================================
  // 🔬 PREMIUM CITATION RENDERER (SENTENCE-LEVEL GRANULAR HIGHLIGHTING)
  // =====================================================================
  const renderMessageText = (text, citations = []) => {
    // Graceful bypass for simple introductory/fallback text or error messages
    if (!citations || citations.length === 0) {
      return <p className="text-slate-100 leading-relaxed whitespace-pre-wrap">{text}</p>;
    }

    // Split paragraphs, then split sentences keeping terminal punctuation
    const paragraphs = text.split("\n");

    return paragraphs.map((paragraph, pIdx) => {
      // Split sentences cleanly keeping punctuation delimiters
      const sentences = paragraph.match(/[^.!?]+[.!?]+(\s|$)/g) || [paragraph];

      return (
        <p key={pIdx} className="text-slate-100 leading-relaxed mb-3 last:mb-0">
          {sentences.map((sentence, sIdx) => {
            if (!sentence.trim()) return null;

            // Fuzzy normalized sentence matching against citation claims
            const cleanSentence = sentence.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
            
            const match = citations.find((c) => {
              const cleanCitation = c.sentence.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
              return (
                cleanSentence === cleanCitation ||
                cleanSentence.includes(cleanCitation) ||
                cleanCitation.includes(cleanSentence)
              );
            });

            if (match) {
              const isHighlighted = hoveredCitation === match.chunk_id;
              return (
                <span
                  key={sIdx}
                  className={`relative inline px-1 rounded cursor-help transition-all duration-200 ${
                    isHighlighted
                      ? "bg-emerald-300/40 text-white border-b-2 border-emerald-400"
                      : "bg-emerald-500/10 hover:bg-emerald-500/25 border-b border-emerald-500/30 text-emerald-100"
                  }`}
                  onMouseEnter={() => setHoveredCitation(match.chunk_id)}
                  onMouseLeave={() => setHoveredCitation(null)}
                >
                  {sentence}
                  {/* Premium CSS Tooltip positioned absolutely above the sentence */}
                  <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block pointer-events-none w-56 bg-slate-950 text-white text-[10px] p-2 rounded shadow-2xl z-50 border border-slate-800">
                    <span className="text-emerald-400 font-semibold block mb-1">✓ NLI Entailment Verified</span>
                    <span className="font-mono text-slate-400 break-all">{match.chunk_id}</span>
                  </span>
                </span>
              );
            }

            return <span key={sIdx}>{sentence}</span>;
          })}
        </p>
      );
    });
  };

  // Get active citations list for the last bot message
  const activeCitations = [...messages]
    .reverse()
    .find((m) => m.sender === "bot" && m.citations && m.citations.length > 0)?.citations || [];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      
      {/* =====================================================================
          🎛️ SIDEBAR: ACTIVE DOCUMENT CITATIONS (BIDIRECTIONAL HOVER SYSTEM)
          ===================================================================== */}
      <aside className="w-80 border-r border-slate-800/80 bg-slate-900/50 flex flex-col hidden lg:flex">
        <header className="p-5 border-b border-slate-800 flex items-center">
          <DatabaseIcon />
          <h2 className="font-semibold text-sm tracking-wide uppercase text-slate-300">Verified Citations</h2>
        </header>
        <main className="flex-1 overflow-y-auto p-4 space-y-3">
          {activeCitations.length === 0 ? (
            <div className="text-center text-xs text-slate-500 mt-20 px-4">
              Ask an in-domain medical query to see live-verified document chunk citations mapped sentence-by-sentence.
            </div>
          ) : (
            activeCitations.map((cit, idx) => {
              const isHovered = hoveredCitation === cit.chunk_id;
              return (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border transition-all duration-200 cursor-default ${
                    isHovered
                      ? "bg-slate-800 border-emerald-500 shadow-md transform scale-[1.02]"
                      : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                  }`}
                  onMouseEnter={() => setHoveredCitation(cit.chunk_id)}
                  onMouseLeave={() => setHoveredCitation(null)}
                >
                  <header className="flex items-center text-[10px] font-mono text-emerald-400 font-semibold mb-2">
                    <CitationIcon />
                    {cit.chunk_id}
                  </header>
                  <p className="text-xs text-slate-300 line-clamp-3 leading-relaxed italic bg-slate-950/40 p-2 rounded border border-slate-800/50">
                    "{cit.sentence}"
                  </p>
                </div>
              );
            })
          )}
        </main>
      </aside>

      {/* =====================================================================
          💬 MAIN CHAT AREA
          ===================================================================== */}
      <main className="flex-1 flex flex-col h-full bg-slate-950 relative">
        
        {/* Sleek Premium Header */}
        <header className="h-16 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between px-6 z-10">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
              <BotIcon />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-wide text-white">AGENTIC SELF-CORRECTING RAG</h1>
              <p className="text-[10px] text-emerald-500 font-mono tracking-wider font-semibold">Active Model: Llama-3 + NLI Critic</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-[10px] text-emerald-400 font-mono font-semibold">FastAPI Connected</span>
          </div>
        </header>

        {/* Messages Log Container */}
        <section className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, index) => {
              const isBot = msg.sender === "bot";
              return (
                <div key={index} className={`flex space-x-4 ${isBot ? "justify-start" : "justify-end"}`}>
                  
                  {isBot && (
                    <div className="w-8 h-8 rounded-full bg-emerald-950 border border-emerald-800/50 flex items-center justify-center flex-shrink-0 shadow-inner">
                      <BotIcon />
                    </div>
                  )}

                  <div className={`max-w-2xl px-5 py-3.5 rounded-2xl shadow-lg border ${
                    isBot 
                      ? "bg-slate-900/70 border-slate-800 text-slate-100" 
                      : "bg-emerald-600 border-emerald-500 text-white ml-auto"
                  }`}>
                    {isBot ? (
                      renderMessageText(msg.text, msg.citations)
                    ) : (
                      <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>
                    )}
                  </div>

                  {!isBot && (
                    <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700/50 flex items-center justify-center flex-shrink-0">
                      <UserIcon />
                    </div>
                  )}

                </div>
              );
            })}

            {/* Pulsing loading placeholder bubble */}
            {loading && (
              <div className="flex space-x-4 justify-start">
                <div className="w-8 h-8 rounded-full bg-emerald-950 border border-emerald-800/50 flex items-center justify-center flex-shrink-0 shadow-inner">
                  <BotIcon />
                </div>
                <div className="max-w-2xl px-5 py-4 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-lg text-slate-300">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-slate-500 mr-2 font-mono">LangGraph executing...</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </section>

        {/* Floating, glowing message input panel */}
        <footer className="p-6 border-t border-slate-800/60 bg-slate-900/10 backdrop-blur-md">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSend} className="relative flex items-center rounded-xl bg-slate-900/90 border border-slate-800/80 shadow-2xl focus-within:border-emerald-500/50 focus-within:shadow-[0_0_20px_rgba(16,185,129,0.06)] transition-all duration-300 p-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a biomedical question (e.g. BRCA1 gene, p53 role, cystic fibrosis mutations)..."
                disabled={loading}
                className="w-full bg-transparent border-none text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-0 text-sm px-4 py-3"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 flex-shrink-0 ${
                  input.trim() && !loading
                    ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_4px_10px_rgba(16,185,129,0.2)] hover:scale-[1.03]"
                    : "bg-slate-800 text-slate-600 cursor-not-allowed"
                }`}
              >
                <SendIcon />
              </button>
            </form>
            <div className="text-center text-[10px] text-slate-600 mt-2 font-mono tracking-wider">
              B.Tech AI Project • Verified Sentence-Level Grounding using Natural Language Inference
            </div>
          </div>
        </footer>

      </main>
    </div>
  );
}
