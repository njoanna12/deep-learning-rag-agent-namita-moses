"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        # TODO: implement
        from dataclasses import replace

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".markdown"}:
            raise ValueError(
                f"Unsupported file type '{suffix}'. This workshop supports Markdown only."
            )

        raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        base_metadata = self._infer_metadata(file_path, metadata_overrides)

        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            chunk_id = VectorStoreManager.generate_chunk_id(
                base_metadata.source, raw["text"]
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=raw["text"],
                    metadata=replace(base_metadata),
                )
            )

        logger.info("Chunked {} into {} chunks", file_path.name, len(chunks))
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        # TODO: implement — iterate and collect, handle per-file errors
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # TODO: implement using langchain_community.document_loaders.PyPDFLoader
        # and langchain.text_splitter.RecursiveCharacterTextSplitter
        raise NotImplementedError

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        # TODO: implement using langchain.text_splitter.MarkdownHeaderTextSplitter
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = file_path.read_text(encoding="utf-8")

        # First split on headers so each section stays together
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        sections = header_splitter.split_text(text)

        # Then split any oversized section into smaller pieces
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        pieces = char_splitter.split_documents(sections)

        results: list[dict] = []
        for piece in pieces:
            body = piece.page_content.strip()
            if not body:
                continue
            header = " > ".join(
                piece.metadata[key]
                for key in ("h1", "h2", "h3")
                if key in piece.metadata
            )
            results.append({"text": body, "header": header})

        return results

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        # TODO: implement filename parsing + override merging
        known_topics = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }
        bonus_topics = {"SOM", "BoltzmannMachine", "GAN"}
        difficulties = {"beginner", "intermediate", "advanced"}

        # Filename convention: <topic>_<difficulty>.md (e.g. lstm_intermediate.md)
        stem = file_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].lower() in difficulties:
            raw_topic, difficulty = parts[0], parts[1].lower()
        else:
            raw_topic, difficulty = stem, "intermediate"

        topic = known_topics.get(raw_topic.lower(), raw_topic)

        metadata = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            related_topics=[],
            is_bonus=topic in bonus_topics,
        )

        # Apply explicit overrides on top of the inferred values
        if overrides:
            for key, value in overrides.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            if "is_bonus" not in overrides:
                metadata.is_bonus = metadata.topic in bonus_topics

        return metadata
