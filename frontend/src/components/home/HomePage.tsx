import { Button } from '@/components/ui/button'

interface HomePageProps {
  loggedIn: boolean
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="py-12">
      <h2 className="text-xl font-bold mb-4">{title}</h2>
      {children}
    </section>
  )
}

export function HomePage({ loggedIn }: HomePageProps) {
  return (
    <div className="mx-auto max-w-5xl px-6">

      {/* Hero */}
      <section className="text-center space-y-4 pt-12 pb-10">
        <h1 className="text-3xl font-bold">Machine Learning Book Recommender</h1>
        <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
          An end‑to‑end project exploring data cleaning, recommendation modeling, and deployment.
        </p>
        <div className="flex flex-wrap justify-center gap-3 pt-2">
          <Button asChild size="lg">
            <a href="/search">Search Books</a>
          </Button>
          <Button asChild size="lg" variant="outline">
            <a href="/profile">Get Recommendations</a>
          </Button>
          {!loggedIn && (
            <Button asChild size="lg" variant="outline">
              <a href="/login">Log In / Sign Up</a>
            </Button>
          )}
          <Button asChild size="lg" variant="outline">
            <a href="/chat">Chat with the Librarian</a>
          </Button>
        </div>
      </section>

      {/* Project note */}
      <p className="text-sm text-muted-foreground italic text-center max-w-2xl mx-auto pb-10">
        <strong className="not-italic text-foreground/70">Project Note:</strong> This app uses the <em>Book‑Crossing</em> dataset as a starting point.
        The raw data contained significant noise and inconsistencies. I implemented extensive cleaning and
        enrichment (e.g., standardized metadata, subject mapping), but a small number of errors can still persist.
        The app exists to showcase the ML pipeline and engineering, not perfect catalog data.
      </p>

      {/* Sectioned content with dividers */}
      <div className="divide-y divide-border">

        {/* What this demonstrates */}
        <Section title="What This Demonstrates">
          <ul className="space-y-2 text-sm list-disc list-inside text-foreground/80">
            <li><strong>End‑to‑end ML build:</strong> data ingestion → cleaning → modeling → serving, with artifact versioning and a quality gate before promotion.</li>
            <li><strong>Recommendation UX:</strong> cold‑start via subjects; warm personalization via collaborative filtering once enough ratings are collected.</li>
            <li><strong>MLOps mindset:</strong> scheduled retraining on a dedicated server, versioned artifacts with SHA-256 checksums, and a Recall@30 gate that blocks regressions from reaching production.</li>
            <li><strong>Production inference:</strong> five isolated model servers (Docker) exposing typed HTTP APIs, decoupling the web app from ML artifacts entirely.</li>
            <li><strong>Multi-agent AI:</strong> a four-stage LangGraph pipeline with streaming output for the AI librarian.</li>
          </ul>
        </Section>

        {/* Behind the scenes */}
        <Section title="Behind the Scenes">
          <div className="grid sm:grid-cols-2 gap-x-10 text-sm text-foreground/80">
            <ul className="space-y-2 list-disc list-inside">
              <li><strong>Data pipeline:</strong> Book‑Crossing base with extensive cleaning and enrichment (standardized metadata, subject mapping).</li>
              <li><strong>Collaborative filtering:</strong> implicit‑feedback <strong>ALS</strong> produces user and item latent factors for personalized candidate generation.</li>
              <li><strong>Content similarity:</strong> attention‑pooled <strong>subject embeddings</strong> (contrastive learning) scored via cosine similarity.</li>
              <li><strong>Popularity:</strong> <strong>Bayesian</strong> smoothing balances rating averages against rating counts to surface reliable signals.</li>
            </ul>
            <ul className="space-y-2 list-disc list-inside mt-2 sm:mt-0">
              <li><strong>Semantic search:</strong> sentence-transformer embeddings over enriched book descriptions, indexed with <strong>FAISS HNSW</strong> for sub-millisecond retrieval.</li>
              <li><strong>Model servers:</strong> five independent <strong>Docker</strong> containers (Embedder, Similarity, ALS, Metadata, Semantic) expose typed HTTP APIs; the web app holds no ML artifacts.</li>
              <li><strong>Training pipeline:</strong> scheduled retraining produces versioned artifacts (version ID + git hash + SHA-256 checksums); a <strong>Recall@30 quality gate</strong> blocks regressions before promotion to production.</li>
            </ul>
          </div>
        </Section>

        {/* Chat with the Librarian */}
        <Section title="Chat with the Librarian (AI Assistant)">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="text-sm">
              <p className="font-semibold mb-2">What it can do</p>
              <ul className="space-y-1 list-disc list-inside text-foreground/80">
                <li>Perform <strong>catalog-grounded semantic search</strong> using enriched subject and description vectors.</li>
                <li>Generate <strong>personalized recommendations</strong> drawing on ALS collaborative filtering, FAISS similarity search, subject embeddings, and Bayesian popularity.</li>
                <li>Explain <strong>why each book fits</strong> your interests with streamed prose and inline citations.</li>
                <li>Answer help and onboarding questions using internal documentation.</li>
                <li>Search the web to suggest <strong>new or trending books</strong> outside the catalog.</li>
              </ul>
            </div>
            <div className="text-sm">
              <p className="font-semibold mb-2">How it works</p>
              <ul className="space-y-1 list-disc list-inside text-foreground/80">
                <li>An LLM-based <em>Router</em> classifies each message and dispatches it to the right branch: recommendation, web search, documentation, or direct reply.</li>
                <li>Recommendation requests run through four agents in sequence: <strong>Planner</strong> selects retrieval tools → <strong>Retrieval</strong> gathers 60–120 candidates via a ReAct loop → <strong>Selection</strong> filters and ranks → <strong>Prose</strong> streams a personalized explanation with inline citations.</li>
                <li>Each stage emits a <strong>status update</strong> in real time so you see progress before the answer arrives.</li>
                <li>Conversation history is stored in <strong>Redis</strong>, giving the assistant multi-turn memory across the session.</li>
              </ul>
            </div>
          </div>
          <div className="mt-4">
            <Button asChild>
              <a href="/chat">Open Chat</a>
            </Button>
          </div>
        </Section>

        {/* Information enrichment */}
        <Section title="Information Enrichment">
          <div className="space-y-3 text-sm text-foreground/80">
            <p>
              Before books are embedded for semantic search, a dedicated <strong>Enrichment Agent</strong> runs as an offline job
              that refines and filters catalog data. Its goal is to transform raw metadata into clean, expressive text
              that better represents each book's content and mood.
            </p>
            <p>
              Each record is reformatted into a compact structure optimized for LLM embeddings—combining{' '}
              <em>title, author, subjects, tone, genre,</em> and <em>vibe</em> into a single enriched description.
              This helps downstream models capture nuance and thematic similarity more effectively.
            </p>
            <p>
              The enrichment pipeline runs on a <strong>Kafka-based workflow</strong> with tiered data quality handling,
              ensuring that books with varying metadata completeness receive appropriate processing depth.
              Two <strong>Spark jobs</strong> handle the output: one ingests enriched data into SQL for quick retrieval,
              while another archives raw enrichment objects in a data lake for versioned storage.
            </p>
            <p>
              Finally, an incremental embedding job encodes new or updated books as they're enriched,
              so the chatbot always works with fresh semantic vectors without needing to rebuild the entire index.
            </p>
          </div>
        </Section>

        {/* How to use */}
        <Section title="How to Use the App">
          <ol className="space-y-3 text-sm list-decimal list-inside text-foreground/80">
            <li>
              <strong>Create Your Profile:</strong>{' '}
              Sign up and select your <strong>favorite genres</strong>. These are used to generate your first set of recommendations.
            </li>
            <li>
              <strong>Explore Initial Recommendations:</strong>{' '}
              Until you've rated at least <strong>10 books</strong>, suggestions are mostly based on your favorite genres. You can update them anytime from your profile.
            </li>
            <li>
              <strong>Browse Popular Books:</strong>{' '}
              Visit the <a href="/search" className="text-primary underline">Search</a> page without entering a title to see books ranked by popularity (using a Bayesian formula). This is a great way to find books you've already read and rate them quickly.
            </li>
            <li>
              <strong>Rate at Least 10 Books:</strong>{' '}
              Once you've rated 10 or more books, the system unlocks <strong>fully personalized recommendations</strong> based on your own preferences.
            </li>
            <li>
              <strong>Keep Exploring and Rating:</strong>{' '}
              The more books you rate, the better your recommendations become. You can also find similar books on any book's detail page.
            </li>
          </ol>
        </Section>

      </div>
    </div>
  )
}
