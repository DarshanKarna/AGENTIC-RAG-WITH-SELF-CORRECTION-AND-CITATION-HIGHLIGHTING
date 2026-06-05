import React, { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Configure pdf.js worker from CDN
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

// =====================================================================
// SVG Icons
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
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-emerald-400 mr-1.5 flex-shrink-0">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-.621-.504-1.125-1.125-1.125H9.75M3 16.061V4.419c0-.847.67-1.57 1.517-1.614 3.78-.2 7.564-.2 11.34 0 .848.044 1.518.767 1.518 1.614v11.642c0 .847-.67 1.57-1.517 1.614-3.78.2-7.564.2-11.34 0A1.516 1.516 0 013 16.061z" />
  </svg>
);

const PaperclipIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32a1.5 1.5 0 01-2.12-2.12l10.517-10.518" />
  </svg>
);

const ChevronLeftIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
  </svg>
);

const ChevronRightIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
  </svg>
);

const UploadCloudIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-slate-500 mb-3">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
  </svg>
);

const DocumentIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 text-emerald-400 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
  </svg>
);

const ZoomInIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6" />
  </svg>
);

const ZoomOutIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6" />
  </svg>
);

const WarningIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-3.5 h-3.5 text-red-400 mr-1 flex-shrink-0 inline">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
  </svg>
);


// =====================================================================
// Helper: Parse chunk_id → page number
// Expected format: docID_p{page}_c{chunk}
// =====================================================================
function parsePageFromChunkId(chunkId) {
  if (!chunkId) return null;
  const match = chunkId.match(/_p(\d+)_c/);
  return match ? parseInt(match[1], 10) : null;
}

// =====================================================================
// Escape special regex characters in a string for safe RegExp usage
// =====================================================================
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// =====================================================================
// Normalize text for fuzzy sentence matching
// =====================================================================
function normalizeForMatch(str) {
  return str.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
}


export default function App() {
  // ----- Chat state -----
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! I am your Self-Correcting RAG assistant. Upload a PDF on the left, then ask me any biomedical question. I will search our local scientific database, self-correct any hallucinations using NLI, and provide verified, sentence-level citations.",
      citations: [],
      // Pipeline Comparison data (null for system messages)
      baseline: null,
      corrected: null
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [hoveredCitation, setHoveredCitation] = useState(null);

  // ----- Pipeline Comparison Mode -----
  // Tracks the active mode PER message index: { [msgIndex]: 'baseline' | 'corrected' }
  const [messageModes, setMessageModes] = useState({});

  // ----- PDF Viewer state -----
  const [activePdfUrl, setActivePdfUrl] = useState(null);
  const [activePdfName, setActivePdfName] = useState("");
  const [numPages, setNumPages] = useState(null);
  const [activePageNumber, setActivePageNumber] = useState(1);
  const [highlightText, setHighlightText] = useState("");
  const [pdfScale, setPdfScale] = useState(1.2);

  // ----- Refs -----
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const pdfViewerFileRef = useRef(null);
  const pdfContainerRef = useRef(null);

  // Auto-scroll to latest messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // =====================================================================
  // Pipeline mode helper — defaults to 'corrected'
  // =====================================================================
  const getMessageMode = (msgIndex) => messageModes[msgIndex] || "corrected";

  const setModeForMessage = (msgIndex, mode) => {
    setMessageModes((prev) => ({ ...prev, [msgIndex]: mode }));
    // Clear PDF highlights when switching modes to avoid stale highlighting
    setHighlightText("");
  };

  // =====================================================================
  // PDF Document handlers
  // =====================================================================
  const onDocumentLoadSuccess = useCallback(({ numPages: total }) => {
    setNumPages(total);
    setActivePageNumber(1);
  }, []);

  const goToPrevPage = () => setActivePageNumber((p) => Math.max(1, p - 1));
  const goToNextPage = () => setActivePageNumber((p) => Math.min(numPages || 1, p + 1));
  const zoomIn = () => setPdfScale((s) => Math.min(3, s + 0.2));
  const zoomOut = () => setPdfScale((s) => Math.max(0.5, s - 0.2));

  // Handle PDF file selection — render in viewer + upload to backend
  const handlePdfFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate PDF
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Error: Only PDF documents are currently supported for dynamic upload.",
          citations: [],
          baseline: null,
          corrected: null
        }
      ]);
      return;
    }

    // Immediately render PDF in viewer
    const objectUrl = URL.createObjectURL(file);
    setActivePdfUrl(objectUrl);
    setActivePdfName(file.name);
    setActivePageNumber(1);
    setHighlightText("");

    // Also upload to FastAPI for ingestion
    setUploading(true);
    setMessages((prev) => [
      ...prev,
      {
        sender: "user",
        text: `[File Upload] Uploading "${file.name}" ...`
      },
      {
        sender: "bot",
        text: `Processing PDF "${file.name}"... Extracting text page-by-page, chunking passages into 500-token windows, and generating local vector embeddings.`,
        citations: [],
        baseline: null,
        corrected: null
      }
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Server returned error status: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Success! Processed and indexed "${file.name}".\n\n• Created: ${data.chunks_count} chunks\n• Indexed in: Local ChromaDB\n\nYou can now ask questions about this document!`,
          citations: [],
          baseline: null,
          corrected: null
        }
      ]);
    } catch (error) {
      console.error("Upload error:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Upload Failed: ${error.message}\n\nThe PDF is still viewable on the left, but it could not be indexed for RAG queries.`,
          citations: [],
          baseline: null,
          corrected: null
        }
      ]);
    } finally {
      setUploading(false);
      if (pdfViewerFileRef.current) pdfViewerFileRef.current.value = "";
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const triggerPdfViewerUpload = () => {
    if (pdfViewerFileRef.current) pdfViewerFileRef.current.click();
  };

  const triggerChatFileUpload = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  // =====================================================================
  // Handle message sending to FastAPI
  // =====================================================================
  const handleSend = async (e) => {
    e.preventDefault();
    const query = input.trim();
    if (!query) return;

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

      // Support both new structured format and legacy flat format
      const hasComparison = data.baseline && data.corrected;

      const baselineData = hasComparison
        ? data.baseline
        : { answer: data.answer || data.draft_answer || "", hallucinated_sentences: [] };

      const correctedData = hasComparison
        ? data.corrected
        : { answer: data.answer || data.draft_answer || "", citations: data.citations || data.verified_citations || [] };

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: correctedData.answer,
          citations: correctedData.citations || [],
          baseline: baselineData,
          corrected: correctedData
        }
      ]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: `Error: Unable to connect to your FastAPI backend. Please make sure api.py is running on http://localhost:8000.\n\nDetails: ${error.message}`,
          citations: [],
          baseline: null,
          corrected: null
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // =====================================================================
  // Citation click handler — navigates PDF to the cited page
  // =====================================================================
  const handleCitationClick = (citation) => {
    const pageNum = parsePageFromChunkId(citation.chunk_id);
    if (pageNum && activePdfUrl) {
      setActivePageNumber(pageNum);
      // Use the actual source chunk text from the PDF, not the LLM answer sentence
      setHighlightText(citation.source_text || citation.sentence);
    }
  };

  // =====================================================================
  // customTextRenderer — highlights matching text on the PDF page
  // Uses contiguous phrase / n-gram matching instead of individual words
  // to avoid highlighting random words scattered across the page.
  // =====================================================================
  const makeTextRenderer = useCallback(
    (textItem) => {
      if (!highlightText || !highlightText.trim()) {
        return textItem.str;
      }

      try {
        const spanText = textItem.str;
        if (!spanText.trim()) return spanText;

        // --- Strategy: build contiguous n-grams from the source text ---
        // The PDF text layer splits content into many small spans, so a
        // long source passage will never appear as one textItem. Instead
        // we extract sliding-window n-grams of 4-7 words from the source
        // text and try to find exact substring matches inside this span.
        const sourceWords = highlightText.trim().split(/\s+/);
        const minGram = Math.min(4, sourceWords.length);
        const maxGram = Math.min(8, sourceWords.length);

        // Collect all unique n-gram strings
        const ngrams = new Set();
        for (let n = maxGram; n >= minGram; n--) {
          for (let i = 0; i <= sourceWords.length - n; i++) {
            ngrams.add(sourceWords.slice(i, i + n).join(" "));
          }
        }

        // Try to match each n-gram as a contiguous substring (case-insensitive)
        let result = spanText;
        let matched = false;

        for (const gram of ngrams) {
          const escaped = escapeRegex(gram);
          const pattern = new RegExp(`(${escaped})`, "gi");
          if (pattern.test(result)) {
            result = result.replace(
              pattern,
              '<mark class="bg-yellow-300 rounded-sm" style="background-color: rgba(253, 224, 71, 0.5); color: transparent; padding: 1px 0;">$1</mark>'
            );
            matched = true;
          }
        }

        if (matched) return result;
      } catch (err) {
        console.warn("Highlight regex error:", err);
      }

      return textItem.str;
    },
    [highlightText]
  );

  // =====================================================================
  // Pipeline Toggle component (rendered inside bot chat bubbles)
  // =====================================================================
  const PipelineToggle = ({ msgIndex }) => {
    const mode = getMessageMode(msgIndex);
    return (
      <div className="flex items-center rounded-lg bg-slate-800/80 border border-slate-700/60 p-0.5 mb-3">
        <button
          onClick={() => setModeForMessage(msgIndex, "baseline")}
          className={`flex-1 text-[11px] font-semibold px-3 py-1.5 rounded-md transition-all duration-200 ${
            mode === "baseline"
              ? "bg-red-500/20 text-red-300 border border-red-500/40 shadow-[0_0_8px_rgba(239,68,68,0.1)]"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          <span className="mr-1.5">⚠</span>Baseline RAG
        </button>
        <button
          onClick={() => setModeForMessage(msgIndex, "corrected")}
          className={`flex-1 text-[11px] font-semibold px-3 py-1.5 rounded-md transition-all duration-200 ${
            mode === "corrected"
              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-[0_0_8px_rgba(16,185,129,0.1)]"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          <span className="mr-1.5">✓</span>Self-Corrected RAG
        </button>
      </div>
    );
  };

  // =====================================================================
  // Render BASELINE text — flags hallucinated sentences with red underline
  // =====================================================================
  const renderBaselineText = (text, hallucinated = []) => {
    if (!text) return null;

    const paragraphs = text.split("\n");
    const normalizedHallucinated = hallucinated.map(normalizeForMatch);

    return paragraphs.map((paragraph, pIdx) => {
      if (!paragraph.trim()) return null;
      const sentences = paragraph.match(/[^.!?]+[.!?]+(\s|$)/g) || [paragraph];

      return (
        <p key={pIdx} className="text-slate-100 leading-relaxed mb-3 last:mb-0">
          {sentences.map((sentence, sIdx) => {
            if (!sentence.trim()) return null;

            const normalizedSentence = normalizeForMatch(sentence);

            const isHallucinated = normalizedHallucinated.some(
              (h) =>
                normalizedSentence === h ||
                normalizedSentence.includes(h) ||
                h.includes(normalizedSentence)
            );

            if (isHallucinated) {
              return (
                <span
                  key={sIdx}
                  className="relative inline px-0.5 border-b-2 border-red-500 bg-red-500/10 text-red-200 rounded-sm cursor-help transition-all duration-200 hover:bg-red-500/20"
                  title="⚠ Hallucination: This sentence failed NLI entailment verification against the source documents."
                >
                  <WarningIcon />
                  {sentence}
                </span>
              );
            }

            return <span key={sIdx}>{sentence}</span>;
          })}
        </p>
      );
    });
  };

  // =====================================================================
  // Render CORRECTED text — green hoverable citation spans
  // =====================================================================
  const renderCorrectedText = (text, citations = []) => {
    if (!citations || citations.length === 0) {
      return <p className="text-slate-100 leading-relaxed whitespace-pre-wrap">{text}</p>;
    }

    const paragraphs = text.split("\n");

    return paragraphs.map((paragraph, pIdx) => {
      const sentences = paragraph.match(/[^.!?]+[.!?]+(\s|$)/g) || [paragraph];

      return (
        <p key={pIdx} className="text-slate-100 leading-relaxed mb-3 last:mb-0">
          {sentences.map((sentence, sIdx) => {
            if (!sentence.trim()) return null;

            const cleanSentence = normalizeForMatch(sentence);

            const match = citations.find((c) => {
              const cleanCitation = normalizeForMatch(c.sentence);
              return (
                cleanSentence === cleanCitation ||
                cleanSentence.includes(cleanCitation) ||
                cleanCitation.includes(cleanSentence)
              );
            });

            if (match) {
              const isHighlighted = hoveredCitation === match.chunk_id;
              const hasPdfPage = activePdfUrl && parsePageFromChunkId(match.chunk_id);
              return (
                <span
                  key={sIdx}
                  className={`relative inline px-1 rounded transition-all duration-200 ${
                    hasPdfPage ? "cursor-pointer" : "cursor-help"
                  } ${
                    isHighlighted
                      ? "bg-emerald-300/40 text-white border-b-2 border-emerald-400"
                      : "bg-emerald-500/10 hover:bg-emerald-500/25 border-b border-emerald-500/30 text-emerald-100"
                  }`}
                  onMouseEnter={() => setHoveredCitation(match.chunk_id)}
                  onMouseLeave={() => setHoveredCitation(null)}
                  onClick={() => handleCitationClick(match)}
                  title={hasPdfPage ? `Click to view page ${parsePageFromChunkId(match.chunk_id)} in PDF` : match.chunk_id}
                >
                  {sentence}
                  {hasPdfPage && (
                    <span className="ml-1 inline-flex items-center text-[9px] bg-emerald-600/30 text-emerald-300 px-1 rounded font-mono">
                      p{parsePageFromChunkId(match.chunk_id)}
                    </span>
                  )}
                </span>
              );
            }

            return <span key={sIdx}>{sentence}</span>;
          })}
        </p>
      );
    });
  };

  // =====================================================================
  // Render message content based on active pipeline mode
  // =====================================================================
  const renderMessageContent = (msg, msgIndex) => {
    const hasComparison = msg.baseline && msg.corrected;

    // For system/info messages without comparison data, render plain text
    if (!hasComparison) {
      return renderCorrectedText(msg.text, msg.citations);
    }

    const mode = getMessageMode(msgIndex);

    if (mode === "baseline") {
      return renderBaselineText(msg.baseline.answer, msg.baseline.hallucinated_sentences);
    } else {
      return renderCorrectedText(msg.corrected.answer, msg.corrected.citations);
    }
  };

  // =====================================================================
  // Render
  // =====================================================================
  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">

      {/* ==============================================================
          LEFT PANE: PDF DOCUMENT VIEWER (50%)
          ============================================================== */}
      <div className="w-1/2 flex flex-col border-r border-slate-800/80 bg-slate-900/30">

        {/* PDF Viewer Header */}
        <header className="h-14 border-b border-slate-800/80 bg-slate-900/60 flex items-center justify-between px-4 flex-shrink-0">
          <div className="flex items-center min-w-0">
            <DocumentIcon />
            <span className="text-sm font-semibold text-slate-200 truncate">
              {activePdfName || "Document Viewer"}
            </span>
          </div>

          <div className="flex items-center space-x-2 flex-shrink-0">
            {/* Hidden file input */}
            <input
              type="file"
              ref={pdfViewerFileRef}
              onChange={handlePdfFileSelect}
              accept="application/pdf"
              className="hidden"
            />
            <button
              onClick={triggerPdfViewerUpload}
              disabled={uploading}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30 hover:border-emerald-500/50 transition-all duration-200 disabled:opacity-40"
            >
              <PaperclipIcon />
              <span>{uploading ? "Processing..." : "Upload PDF"}</span>
            </button>
          </div>
        </header>

        {/* PDF Content Area */}
        {!activePdfUrl ? (
          /* Empty state — drop zone prompt */
          <div className="flex-1 flex items-center justify-center">
            <div
              onClick={triggerPdfViewerUpload}
              className="cursor-pointer flex flex-col items-center text-center px-8 py-12 border-2 border-dashed border-slate-700/60 rounded-2xl hover:border-emerald-500/40 hover:bg-slate-900/40 transition-all duration-300 max-w-sm"
            >
              <UploadCloudIcon />
              <p className="text-sm font-medium text-slate-400 mb-1">Upload a PDF to get started</p>
              <p className="text-xs text-slate-600">
                Click here or use the upload button above. Your document will appear here for visual reference alongside the RAG chat.
              </p>
            </div>
          </div>
        ) : (
          /* PDF rendering area */
          <div className="flex-1 flex flex-col min-h-0">

            {/* Page controls toolbar */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-900/80 border-b border-slate-800/60 flex-shrink-0">
              <div className="flex items-center space-x-1">
                <button
                  onClick={goToPrevPage}
                  disabled={activePageNumber <= 1}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
                  title="Previous page"
                >
                  <ChevronLeftIcon />
                </button>
                <span className="text-xs font-mono text-slate-400 min-w-[80px] text-center">
                  Page {activePageNumber} / {numPages || "—"}
                </span>
                <button
                  onClick={goToNextPage}
                  disabled={activePageNumber >= (numPages || 1)}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
                  title="Next page"
                >
                  <ChevronRightIcon />
                </button>
              </div>

              <div className="flex items-center space-x-1">
                <button
                  onClick={zoomOut}
                  disabled={pdfScale <= 0.5}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                  title="Zoom out"
                >
                  <ZoomOutIcon />
                </button>
                <span className="text-[10px] font-mono text-slate-500 min-w-[40px] text-center">
                  {Math.round(pdfScale * 100)}%
                </span>
                <button
                  onClick={zoomIn}
                  disabled={pdfScale >= 3}
                  className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                  title="Zoom in"
                >
                  <ZoomInIcon />
                </button>
              </div>

              {highlightText && (
                <button
                  onClick={() => setHighlightText("")}
                  className="text-[10px] px-2 py-1 rounded bg-yellow-500/15 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/25 transition-colors"
                >
                  Clear Highlight
                </button>
              )}
            </div>

            {/* Scrollable PDF render */}
            <div ref={pdfContainerRef} className="flex-1 overflow-auto flex justify-center bg-slate-950/50 p-4">
              <Document
                file={activePdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={
                  <div className="flex items-center justify-center h-64">
                    <div className="flex flex-col items-center space-y-3">
                      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                      <span className="text-xs text-slate-500 font-mono">Loading PDF...</span>
                    </div>
                  </div>
                }
                error={
                  <div className="flex items-center justify-center h-64 text-red-400 text-sm">
                    Failed to load PDF. Please try a different file.
                  </div>
                }
              >
                <Page
                  key={`page_${activePageNumber}_${highlightText}`}
                  pageNumber={activePageNumber}
                  scale={pdfScale}
                  customTextRenderer={makeTextRenderer}
                  className="shadow-2xl rounded-lg overflow-hidden"
                  loading={
                    <div className="flex items-center justify-center h-64">
                      <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                  }
                />
              </Document>
            </div>
          </div>
        )}
      </div>


      {/* ==============================================================
          RIGHT PANE: CHAT INTERFACE (50%)
          ============================================================== */}
      <div className="w-1/2 flex flex-col h-full bg-slate-950 relative">

        {/* Chat Header */}
        <header className="h-14 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between px-5 flex-shrink-0 z-10">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
              <BotIcon />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-wide text-white">AGENTIC SELF-CORRECTING RAG</h1>
              <p className="text-[10px] text-emerald-500 font-mono tracking-wider font-semibold">Llama-3 + NLI Critic • Pipeline Comparison</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-[10px] text-emerald-400 font-mono font-semibold">FastAPI Connected</span>
          </div>
        </header>

        {/* Messages Container */}
        <section className="flex-1 overflow-y-auto px-5 py-6 space-y-5">
          <div className="space-y-5">
            {messages.map((msg, index) => {
              const isBot = msg.sender === "bot";
              const hasComparison = msg.baseline && msg.corrected;
              const mode = getMessageMode(index);

              return (
                <div key={index} className={`flex space-x-3 ${isBot ? "justify-start" : "justify-end"}`}>

                  {isBot && (
                    <div className="w-7 h-7 rounded-full bg-emerald-950 border border-emerald-800/50 flex items-center justify-center flex-shrink-0 shadow-inner mt-0.5">
                      <BotIcon />
                    </div>
                  )}

                  <div className={`max-w-[85%] px-4 py-3 rounded-2xl shadow-lg border ${
                    isBot
                      ? "bg-slate-900/70 border-slate-800 text-slate-100"
                      : "bg-emerald-600 border-emerald-500 text-white ml-auto"
                  }`}>
                    {isBot ? (
                      <>
                        {/* Pipeline Comparison Toggle — only for messages with comparison data */}
                        {hasComparison && <PipelineToggle msgIndex={index} />}

                        {/* Mode badge indicator */}
                        {hasComparison && (
                          <div className={`inline-flex items-center text-[10px] font-mono px-2 py-0.5 rounded-full mb-2 ${
                            mode === "baseline"
                              ? "bg-red-500/15 text-red-400 border border-red-500/30"
                              : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                          }`}>
                            {mode === "baseline" ? "⚠ Unverified baseline output" : "✓ NLI-verified output"}
                          </div>
                        )}

                        {/* Render content based on active mode */}
                        {renderMessageContent(msg, index)}

                        {/* Citation chips — only shown in corrected mode */}
                        {hasComparison && mode === "corrected" && msg.corrected.citations && msg.corrected.citations.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap gap-1.5">
                            {msg.corrected.citations.map((cit, cIdx) => {
                              const pageNum = parsePageFromChunkId(cit.chunk_id);
                              return (
                                <button
                                  key={cIdx}
                                  onClick={() => handleCitationClick(cit)}
                                  onMouseEnter={() => setHoveredCitation(cit.chunk_id)}
                                  onMouseLeave={() => setHoveredCitation(null)}
                                  className={`inline-flex items-center text-[10px] font-mono px-2 py-1 rounded-md border transition-all duration-200 ${
                                    hoveredCitation === cit.chunk_id
                                      ? "bg-emerald-500/20 border-emerald-500 text-emerald-300 scale-105"
                                      : "bg-slate-800/60 border-slate-700/50 text-slate-400 hover:border-emerald-500/40 hover:text-emerald-400"
                                  } ${activePdfUrl && pageNum ? "cursor-pointer" : "cursor-default"}`}
                                  title={cit.sentence}
                                >
                                  <CitationIcon />
                                  {cit.chunk_id}
                                </button>
                              );
                            })}
                          </div>
                        )}

                        {/* Hallucination count indicator — only in baseline mode */}
                        {hasComparison && mode === "baseline" && msg.baseline.hallucinated_sentences && msg.baseline.hallucinated_sentences.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-red-800/30">
                            <div className="inline-flex items-center text-[10px] font-mono px-2 py-1 rounded-md bg-red-500/10 border border-red-500/30 text-red-400">
                              <WarningIcon />
                              <span className="ml-1">{msg.baseline.hallucinated_sentences.length} sentence(s) flagged as potential hallucinations by NLI Critic</span>
                            </div>
                          </div>
                        )}

                        {/* Legacy citation chips for messages without comparison data */}
                        {!hasComparison && msg.citations && msg.citations.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap gap-1.5">
                            {msg.citations.map((cit, cIdx) => {
                              const pageNum = parsePageFromChunkId(cit.chunk_id);
                              return (
                                <button
                                  key={cIdx}
                                  onClick={() => handleCitationClick(cit)}
                                  onMouseEnter={() => setHoveredCitation(cit.chunk_id)}
                                  onMouseLeave={() => setHoveredCitation(null)}
                                  className={`inline-flex items-center text-[10px] font-mono px-2 py-1 rounded-md border transition-all duration-200 ${
                                    hoveredCitation === cit.chunk_id
                                      ? "bg-emerald-500/20 border-emerald-500 text-emerald-300 scale-105"
                                      : "bg-slate-800/60 border-slate-700/50 text-slate-400 hover:border-emerald-500/40 hover:text-emerald-400"
                                  } ${activePdfUrl && pageNum ? "cursor-pointer" : "cursor-default"}`}
                                  title={cit.sentence}
                                >
                                  <CitationIcon />
                                  {cit.chunk_id}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </>
                    ) : (
                      <p className="leading-relaxed whitespace-pre-wrap text-sm">{msg.text}</p>
                    )}
                  </div>

                  {!isBot && (
                    <div className="w-7 h-7 rounded-full bg-slate-800 border border-slate-700/50 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <UserIcon />
                    </div>
                  )}
                </div>
              );
            })}

            {/* Loading indicator */}
            {loading && (
              <div className="flex space-x-3 justify-start">
                <div className="w-7 h-7 rounded-full bg-emerald-950 border border-emerald-800/50 flex items-center justify-center flex-shrink-0 shadow-inner">
                  <BotIcon />
                </div>
                <div className="px-4 py-3 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-lg text-slate-300">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-slate-500 mr-1 font-mono">LangGraph executing...</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "0ms" }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "150ms" }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "300ms" }}></span>
                  </div>
                </div>
              </div>
            )}

            {/* Uploading indicator */}
            {uploading && (
              <div className="flex space-x-3 justify-start">
                <div className="w-7 h-7 rounded-full bg-emerald-950 border border-emerald-800/50 flex items-center justify-center flex-shrink-0 shadow-inner">
                  <BotIcon />
                </div>
                <div className="px-4 py-3 rounded-2xl bg-slate-900/70 border border-slate-800 shadow-lg text-slate-300">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-slate-500 mr-1 font-mono">Processing PDF...</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "0ms" }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "150ms" }}></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" style={{ animationDelay: "300ms" }}></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </section>

        {/* Message Input */}
        <footer className="p-4 border-t border-slate-800/60 bg-slate-900/10 backdrop-blur-md flex-shrink-0">
          <div className="max-w-full mx-auto">
            {/* Hidden file input for chat paperclip */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handlePdfFileSelect}
              accept="application/pdf"
              className="hidden"
            />

            <form onSubmit={handleSend} className="relative flex items-center rounded-xl bg-slate-900/90 border border-slate-800/80 shadow-2xl focus-within:border-emerald-500/50 focus-within:shadow-[0_0_20px_rgba(16,185,129,0.06)] transition-all duration-300 p-2">
              <button
                type="button"
                onClick={triggerChatFileUpload}
                disabled={loading || uploading}
                className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all duration-200 flex-shrink-0 mr-1 ${
                  loading || uploading
                    ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                    : "bg-slate-800/80 hover:bg-slate-800 text-slate-400 hover:text-slate-100 hover:scale-[1.03]"
                }`}
                title="Upload PDF"
              >
                <PaperclipIcon />
              </button>

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question..."
                disabled={loading || uploading}
                className="w-full bg-transparent border-none text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-0 text-sm px-3 py-2.5"
              />
              <button
                type="submit"
                disabled={loading || !input.trim() || uploading}
                className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all duration-200 flex-shrink-0 ${
                  input.trim() && !loading && !uploading
                    ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_4px_10px_rgba(16,185,129,0.2)] hover:scale-[1.03]"
                    : "bg-slate-800 text-slate-600 cursor-not-allowed"
                }`}
              >
                <SendIcon />
              </button>
            </form>
            <div className="text-center text-[10px] text-slate-600 mt-2 font-mono tracking-wider">
              B.Tech AI Project • Pipeline Comparison: Baseline RAG vs. Self-Corrected NLI-Verified Output
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
