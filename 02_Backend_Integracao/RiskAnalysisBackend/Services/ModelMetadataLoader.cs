using System.Text.Json;
using RiskAnalysisBackend.Configuration;
using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public static class ModelMetadataLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    public static ModelMetadata Carregar(MLOptions options)
    {
        string caminho = MLOptions.ResolverCaminho(options.MetadataPath);

        if (!File.Exists(caminho))
            throw new FileNotFoundException($"Arquivo de metadados do modelo não encontrado em: {caminho}");

        ModelMetadata metadata;
        try
        {
            metadata = JsonSerializer.Deserialize<ModelMetadata>(File.ReadAllText(caminho), JsonOptions)
                       ?? throw new InvalidOperationException("O arquivo de metadados está vazio.");
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"model_metadata.json inválido: {ex.Message}", ex);
        }

        Validar(metadata);
        return metadata;
    }

    private static void Validar(ModelMetadata metadata)
    {
        if (metadata.FeatureNames.Count == 0)
            throw new InvalidOperationException("model_metadata.json não define nenhum feature_names.");

        if (metadata.FeatureNames.Any(string.IsNullOrWhiteSpace))
            throw new InvalidOperationException("feature_names contém nomes vazios.");

        var duplicados = metadata.FeatureNames
            .GroupBy(n => n, StringComparer.OrdinalIgnoreCase)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();
        if (duplicados.Count > 0)
            throw new InvalidOperationException($"feature_names contém nomes duplicados: {string.Join(", ", duplicados)}.");

        var semMediana = metadata.FeatureNames.Where(n => !metadata.Medians.ContainsKey(n)).ToList();
        if (semMediana.Count > 0)
            throw new InvalidOperationException($"Faltam medianas de treino para: {string.Join(", ", semMediana)}.");

        if (!(metadata.ThresholdMedioRisco > 0f
              && metadata.ThresholdMedioRisco < metadata.ThresholdAltoRisco
              && metadata.ThresholdAltoRisco <= 1f))
            throw new InvalidOperationException("Os limiares de risco devem respeitar 0 < médio < alto <= 1.");
    }
}
