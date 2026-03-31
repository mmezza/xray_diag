"""
Claude API service for X-ray image analysis.

Uses Claude Opus 4.6 with vision capabilities and adaptive thinking
to provide detailed radiological assessments.
"""

import anthropic
from config import settings

SYSTEM_PROMPT = """Você é um assistente especializado em análise de imagens radiológicas,
treinado para apoiar profissionais de saúde na interpretação de raios-X.

Sua função é identificar achados radiológicos relevantes, descrever alterações anatômicas
e fornecer uma avaliação estruturada da imagem. Você **não emite diagnósticos clínicos finais**
— suas análises são suporte à decisão e devem sempre ser validadas por um radiologista.

Ao analisar uma imagem, seja preciso, técnico e organizado. Use terminologia radiológica
adequada. Se a qualidade da imagem limitar a análise, informe claramente."""

ANALYSIS_PROMPT = """Analise esta imagem de raio-X e forneça um relatório radiológico estruturado.

Retorne sua análise EXATAMENTE no seguinte formato JSON (sem markdown, apenas JSON puro):

{
  "image_quality": {
    "score": "<Boa|Regular|Ruim>",
    "notes": "<observações sobre posicionamento, exposição, artefatos>"
  },
  "region": "<Tórax|Abdome|Coluna|Extremidade|Crânio|Pelve|Outro>",
  "laterality": "<PA|AP|Lateral|Oblíqua|N/A>",
  "findings": [
    {
      "location": "<localização anatômica>",
      "description": "<descrição técnica do achado>",
      "severity": "<Normal|Leve|Moderado|Grave>"
    }
  ],
  "normal_structures": ["<lista de estruturas com aparência normal>"],
  "areas_of_concern": [
    {
      "area": "<área específica>",
      "observation": "<descrição da alteração>",
      "significance": "<significado clínico provável>"
    }
  ],
  "risk_level": "<Normal|Baixo|Moderado|Alto|Crítico>",
  "summary": "<resumo em 2-3 frases da análise geral>",
  "recommendations": ["<lista de recomendações>"],
  "confidence": <número entre 0 e 100>,
  "disclaimer": "Esta análise é gerada por IA para suporte à decisão clínica e deve ser revisada por um radiologista certificado."
}

Arquivo analisado: {filename}

IMPORTANTE: Retorne APENAS o JSON, sem texto adicional antes ou depois."""


class ClaudeService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    async def analyze_xray(
        self, image_b64: str, media_type: str, original_filename: str = "imagem.jpg"
    ) -> dict:
        """
        Analyze an X-ray image using Claude's vision capabilities.

        Uses adaptive thinking for thorough radiological assessment.
        Returns a structured diagnostic report as a dict.
        """
        prompt = ANALYSIS_PROMPT.format(filename=original_filename)

        if media_type == "application/pdf":
            message_content = [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]

        with self.client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message_content}],
        ) as stream:
            response = stream.get_final_message()

        raw_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()

        import json

        try:
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(
                    l for l in lines if not l.startswith("```")
                ).strip()
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            result = _parse_fallback(raw_text)

        result["raw_response"] = raw_text
        result["model"] = "claude-opus-4-6"
        result["thinking_tokens"] = sum(
            len(b.thinking) if hasattr(b, "thinking") else 0
            for b in response.content
            if b.type == "thinking"
        )
        return result

    def build_search_text(self, diagnosis: dict) -> str:
        """Build a searchable text representation of the diagnosis."""
        parts = [
            diagnosis.get("region", ""),
            diagnosis.get("summary", ""),
            diagnosis.get("risk_level", ""),
        ]
        for f in diagnosis.get("findings", []):
            parts.append(f.get("description", ""))
        for c in diagnosis.get("areas_of_concern", []):
            parts.append(c.get("observation", ""))
        return " ".join(filter(None, parts))


def _parse_fallback(text: str) -> dict:
    """Return a minimal diagnostic dict when JSON parsing fails."""
    return {
        "image_quality": {"score": "Regular", "notes": "Não foi possível determinar"},
        "region": "Desconhecido",
        "laterality": "N/A",
        "findings": [],
        "normal_structures": [],
        "areas_of_concern": [],
        "risk_level": "Indeterminado",
        "summary": text[:500] if text else "Análise não disponível",
        "recommendations": ["Revisar manualmente"],
        "confidence": 0,
        "disclaimer": "Esta análise é gerada por IA para suporte à decisão clínica e deve ser revisada por um radiologista certificado.",
        "parse_error": True,
    }
