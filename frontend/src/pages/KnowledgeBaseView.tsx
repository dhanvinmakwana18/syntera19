import React, { useState, useEffect } from 'react';
import { UploadCloud, File, Trash2, RefreshCw } from 'lucide-react';
import { uploadDocument, getDocuments } from '../api/client';

export default function KnowledgeBaseView() {
  const [documents, setDocuments] = useState<any[]>([]);
  const [uploading, setUploading] = useState(false);

  const fetchDocs = async () => {
    try {
      const data = await getDocuments();
      setDocuments(data.documents || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploading(true);
      try {
        await uploadDocument(e.target.files[0]);
        await fetchDocs();
      } catch (error) {
        alert("Upload failed.");
      } finally {
        setUploading(false);
      }
    }
  };

  return (
    <div className="h-full p-8 overflow-y-auto">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl font-bold text-white mb-2">Knowledge Base</h2>
        <p className="text-zinc-400 mb-8">Manage the documents and data sources indexed in the vector store.</p>

        {/* Upload Area */}
        <div className="border-2 border-dashed border-zinc-700/50 rounded-2xl p-10 text-center hover:bg-zinc-900/30 transition-colors mb-10 relative">
          <input 
            type="file" 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            onChange={handleFileChange}
            disabled={uploading}
            accept=".pdf,.txt,.md"
          />
          <div className="flex flex-col items-center justify-center">
            {uploading ? (
              <RefreshCw className="w-10 h-10 text-blue-400 animate-spin mb-4" />
            ) : (
              <UploadCloud className="w-10 h-10 text-zinc-500 mb-4" />
            )}
            <h3 className="text-lg font-semibold text-white mb-1">
              {uploading ? 'Processing & Indexing...' : 'Upload Document'}
            </h3>
            <p className="text-sm text-zinc-500">
              Drag & drop or click to browse. Supports PDF, TXT, MD.
            </p>
          </div>
        </div>

        {/* Document List */}
        <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-2xl overflow-hidden">
          <div className="px-6 py-4 border-b border-zinc-800/80 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">Indexed Sources</h3>
            <span className="text-xs font-mono text-zinc-500">{documents.length} items</span>
          </div>
          <div className="divide-y divide-zinc-800/60">
            {documents.length === 0 && (
              <div className="p-8 text-center text-sm text-zinc-500">
                No documents found. Upload one to begin.
              </div>
            )}
            {documents.map((doc, idx) => (
              <div key={idx} className="px-6 py-4 flex items-center justify-between hover:bg-zinc-800/30 transition-colors">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
                    <File size={16} />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-zinc-200">{doc.name}</div>
                    <div className="text-xs font-mono text-zinc-500 mt-0.5">{doc.size_kb} KB</div>
                  </div>
                </div>
                <button className="p-2 text-zinc-500 hover:text-red-400 transition-colors">
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
