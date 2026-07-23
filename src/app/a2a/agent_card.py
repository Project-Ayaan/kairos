from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    SecurityScheme,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    StringList
)
from app.core.config import settings

# Define the baseline machine-readable A2A AgentCard for Kairos CDSS discovery.
KAIROS_AGENT_CARD = AgentCard(
    name="Kairos CDSS",
    description="Evidence-grounded Clinical Decision Support agent. Answers clinical queries by retrieving and synthesizing PubMed literature alongside other reference material such as policy documents. Returns a synthesized answer with full source citation metadata (source type, PMID, DOI, PMCID, PubMed URL, document name/section, authors, journal, year, relevance score).",
    version="0.1.0",
    capabilities=AgentCapabilities(
        streaming=False,
        push_notifications=False
    ),
    skills=[
        AgentSkill(
            id="clinical_query",
            name="Clinical Evidence Query",
            description="Accepts a clinical or policy question and returns a grounded synthesis with source metadata drawn from PubMed literature and/or reference documents.",
            input_modes=["text"],
            output_modes=["text", "data"],
            tags=["clinical", "evidence", "pubmed", "cdss", "medical", "policy"]
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text", "data"]
)

def get_agent_card() -> AgentCard:
    """Returns the AgentCard, dynamically adding security requirements if A2A_API_KEY is set."""
    card = AgentCard()
    card.CopyFrom(KAIROS_AGENT_CARD)
    
    if settings.a2a_api_key:
        # Enforce Bearer authentication scheme
        card.security_schemes["bearer_auth"].CopyFrom(
            SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="APIKey",
                    description="A2A API key authentication."
                )
            )
        )
        
        # Add to card requirements list
        req = SecurityRequirement()
        req.schemes["bearer_auth"].CopyFrom(StringList(list=[]))
        card.security_requirements.append(req)
        
    return card
