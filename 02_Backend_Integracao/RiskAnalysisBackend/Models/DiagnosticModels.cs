using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace RiskAnalysisBackend.Models;

public class FinancialDataRequest
{
    [RegularExpression(@"^[A-Za-z0-9_\-]{1,64}$", ErrorMessage = "empresaId deve ter de 1 a 64 caracteres (letras, números, '-' ou '_').")]
    public string? EmpresaId { get; set; }

    [StringLength(100, ErrorMessage = "nomeEmpresa deve ter no máximo 100 caracteres.")]
    public string? NomeEmpresa { get; set; }

    [Range(1, 5, ErrorMessage = "horizonteAnos deve estar entre 1 e 5.")]
    public int HorizonteAnos { get; set; } = 1;

    [StringLength(1000, ErrorMessage = "contextoUsuario deve ter no máximo 1000 caracteres.")]
    public string? ContextoUsuario { get; set; }

    public Dictionary<string, double?>? Indicadores { get; set; }
}

public class ModelMetadata
{
    [JsonPropertyName("feature_names")]
    public List<string> FeatureNames { get; set; } = new();

    [JsonPropertyName("feature_descriptions")]
    public Dictionary<string, string> FeatureDescriptions { get; set; } = new();

    [JsonPropertyName("medians")]
    public Dictionary<string, float> Medians { get; set; } = new();

    [JsonPropertyName("threshold_medio_risco")]
    public float ThresholdMedioRisco { get; set; } = 0.35f;

    [JsonPropertyName("threshold_alto_risco")]
    public float ThresholdAltoRisco { get; set; } = 0.65f;
}

public class RiskDiagnosisResponse
{
    public string EmpresaId { get; set; } = string.Empty;
    public string NomeEmpresa { get; set; } = string.Empty;
    public int HorizonteAnos { get; set; }
    public DateTime DataAnalise { get; set; } = DateTime.UtcNow;
    public int ClassePrevista { get; set; }
    public string ClassificacaoRisco { get; set; } = string.Empty;
    public float ProbabilidadeSaudavel { get; set; }
    public float ProbabilidadeInsolvencia { get; set; }
    public string ModoExecucao { get; set; } = string.Empty;
    public float PercentualIndicadoresInformados { get; set; }
    public Dictionary<string, float> IndicadoresProcessados { get; set; } = new();
    public List<string> AtributosImputados { get; set; } = new();
    public List<string> PontosDeAtencao { get; set; } = new();
    public List<string> AlertasValidacao { get; set; } = new();
}

public class ChatDiagnosticContextResponse
{
    public string EmpresaId { get; set; } = string.Empty;
    public string NomeEmpresa { get; set; } = string.Empty;
    public string ClassificacaoRisco { get; set; } = string.Empty;
    public float ProbabilidadeRiscoPercentual { get; set; }
    public string? ContextoInformadoPeloUsuario { get; set; }
    public string ResumoParaPromptLLM { get; set; } = string.Empty;
    public RiskDiagnosisResponse DiagnosticoCompleto { get; set; } = new();
}

public record ErroResponse(string Erro, IReadOnlyList<string>? Detalhes = null);

public record ServiceStatusResponse(
    string Status,
    string ModoExecucao,
    bool ModeloOnnxCarregado,
    int TotalAtributos,
    int DiagnosticosEmMemoria);

public record PredictionResult(int Classe, float ProbabilidadeSaudavel, float ProbabilidadeInsolvencia);

public record PreprocessingResult(
    float[] Vetor,
    Dictionary<string, float> Processados,
    List<string> Imputados,
    List<string> Alertas);
