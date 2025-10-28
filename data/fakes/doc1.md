# HP-RAG Guide

Welcome to the HP-RAG Guide. This guide explains TOC indexing and link filtering.

## Overview

HP-RAG uses a Table of Contents (TOC) and hyperlinks to organize sections. Retrieval favors structured navigation and neighbor sections over heavy ML. The default selection method uses an LLM to filter appropriate links from the TOC.

## Installation

1. Prepare your corpus in Markdown with clear headings.
2. Run the ingestion to build TOC indexing into SQLite.
3. Optionally build a FAISS index for vector baselines.

## Using HP-RAG

### TOC Indexing

TOC indexing analyzes headings (e.g., #, ##, ###) to build hierarchical paths like `doc/section/subsection`. Each section stores its path, title, body, and ordering.

### Link Filtering

The system performs LLM-based link filtering by default. Given a query, the LLM selects relevant section paths from the TOC. You can optionally enable FTS fallback when no LLM is available.

