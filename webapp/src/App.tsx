import { useState } from 'react';
import axios from 'axios';
import './App.css';

interface SearchResult {
  text: string;
  source: string;
  chunk_index: number;
}

interface ApiResponse {
  response: SearchResult[] | string;
}

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[] | string | null>(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const searchModes = {
    BASIC: '/search',
    AI: '/ai-search',
    DIRECT: '/ai-direct'
  };

  const handleSearch = async (endpoint: string) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const { data } = await axios.get<ApiResponse>(`${API_BASE_URL}${endpoint}`, {
        params: { query }
      });

      setResults(data.response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const renderResults = () => {
    if (!results) return null;

    if (typeof results === 'string') {
      return <div className="ai-response">{results}</div>;
    }

    return (
      <div className="search-results">
        {results.map((result, index) => (
          <div key={index} className="result-card">
            <div className="source-info">
              <span className="source">{result.source}</span>
              <span className="chunk">Chunk {result.chunk_index}</span>
            </div>
            <p className="result-text">{result.text}</p>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="container">
      <h1>AI GM Assistant</h1>

      <div className="search-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about RPG rules..."
          className="search-input"
        />

        <div className="button-group">
          <button
            onClick={() => handleSearch(searchModes.BASIC)}
            disabled={loading}
          >
            Basic Search
          </button>
          <button
            onClick={() => handleSearch(searchModes.AI)}
            disabled={loading}
          >
            AI Search
          </button>
          <button
            onClick={() => handleSearch(searchModes.DIRECT)}
            disabled={loading}
          >
            Direct AI
          </button>
        </div>
      </div>

      {loading && <div className="loading">Searching...</div>}
      {error && <div className="error">{error}</div>}
      {renderResults()}
    </div>
  );
}

export default App;