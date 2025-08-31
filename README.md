# Knowledge Graph Generator

A pure Rust library and CLI tool for generating knowledge graphs from web URLs and documents.

## Features

- 🌐 **Web scraping**: Extract content from any web URL
- 📄 **Document processing**: Support for PDF, DOCX, TXT, and Markdown files  
- 🧠 **NLP processing**: Extract entities and relations using pattern matching
- 📊 **Graph representation**: Build structured knowledge graphs with nodes and edges
- 💾 **Export/Import**: JSON serialization for graph persistence
- ⚡ **Async processing**: Built with Tokio for efficient I/O operations
- 🔧 **Pure Rust**: No external dependencies on Python or other languages

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
knowledge-graph = "0.1.0"
```

## Quick Start

### As a Library

```rust
use knowledge_graph::{GraphBuilder, ContentSource};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    
    // Process a web URL
    let source = ContentSource::from_url("https://example.com");
    let graph = builder.process_source(source).await?;
    
    println!("Generated {} nodes and {} edges", 
             graph.node_count(), graph.edge_count());
    
    // Export to JSON
    let json = graph.to_json()?;
    tokio::fs::write("graph.json", json).await?;
    
    Ok(())
}
```

### As a CLI Tool

```bash
# Process a web URL
cargo run -- https://en.wikipedia.org/wiki/Artificial_intelligence

# Process a local document
cargo run -- /path/to/document.pdf

# The tool will generate knowledge_graph.json with the results
```

## Supported Input Types

- **Web URLs**: Any HTTP/HTTPS URL with text content
- **PDF files**: Text extraction from PDF documents
- **DOCX files**: Microsoft Word document processing
- **Text files**: Plain text and Markdown files
- **Raw text**: Direct text input

## Architecture

### Core Components

1. **ContentExtractor**: Handles different input sources and file formats
2. **NlpProcessor**: Extracts entities and relations using pattern matching
3. **KnowledgeGraph**: Graph data structure with nodes and edges
4. **GraphBuilder**: Main orchestrator that combines all components

### Entity Types

- **Person**: Individual people (detected via name patterns)
- **Organization**: Companies, institutions, groups
- **Location**: Places, addresses, geographical entities
- **Date**: Temporal information
- **Number**: Numerical data and statistics
- **Concept**: General concepts and topics

### Relation Types

- **is_a**: Taxonomic relationships
- **works_for**: Employment relationships
- **owns**: Ownership relationships
- **references**: Document references
- **similar_to**: Similarity connections

## API Documentation

### GraphBuilder

The main entry point for creating knowledge graphs:

```rust
let mut builder = GraphBuilder::new();
let graph = builder.process_source(source).await?;
```

### KnowledgeGraph

Core graph operations:

```rust
// Add nodes and edges
let node_id = graph.add_node(node);
let edge_id = graph.add_edge(edge)?;

// Query the graph
let neighbors = graph.neighbors(node_id);
let path = graph.shortest_path(from_node, to_node);

// Serialize/deserialize
let json = graph.to_json()?;
let restored = KnowledgeGraph::from_json(&json)?;
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.rs`: Text processing and entity extraction
- `web_scraping.rs`: Processing web content

## Performance Considerations

- **Memory usage**: Large documents may require significant memory for processing
- **Network requests**: Web scraping includes rate limiting and timeout handling
- **Concurrent processing**: Built on Tokio for async operations

## Error Handling

The library uses custom error types for different failure scenarios:

```rust
match result {
    Err(KnowledgeGraphError::Http(_)) => {
        // Handle network errors
    }
    Err(KnowledgeGraphError::UnsupportedContentType { content_type }) => {
        // Handle unsupported file types
    }
    Err(KnowledgeGraphError::Extraction { message }) => {
        // Handle content extraction errors
    }
    Ok(graph) => {
        // Process successful result
    }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `cargo test` passes
5. Submit a pull request

## License

MIT License - see LICENSE file for details.