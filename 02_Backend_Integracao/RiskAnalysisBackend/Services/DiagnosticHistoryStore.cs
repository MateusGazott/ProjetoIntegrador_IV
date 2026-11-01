using Microsoft.Extensions.Options;
using RiskAnalysisBackend.Configuration;
using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public interface IDiagnosticHistoryStore
{
    int Quantidade { get; }
    void Salvar(ChatDiagnosticContextResponse contexto);
    ChatDiagnosticContextResponse? Obter(string? empresaId);
}

public sealed class DiagnosticHistoryStore : IDiagnosticHistoryStore
{
    private sealed record Entrada(ChatDiagnosticContextResponse Contexto, DateTime CriadoEmUtc);

    private readonly Dictionary<string, Entrada> _itens = new(StringComparer.Ordinal);
    private readonly object _trava = new();
    private readonly int _maximo;
    private readonly TimeSpan _validade;
    private string? _ultimoId;

    public DiagnosticHistoryStore(IOptions<MLOptions> options)
    {
        _maximo = Math.Max(1, options.Value.MaximoDiagnosticosEmMemoria);
        _validade = TimeSpan.FromMinutes(Math.Max(1, options.Value.ValidadeDiagnosticoMinutos));
    }

    public int Quantidade
    {
        get
        {
            lock (_trava)
            {
                RemoverExpirados();
                return _itens.Count;
            }
        }
    }

    public void Salvar(ChatDiagnosticContextResponse contexto)
    {
        lock (_trava)
        {
            _itens[contexto.EmpresaId] = new Entrada(contexto, DateTime.UtcNow);
            _ultimoId = contexto.EmpresaId;

            RemoverExpirados();
            while (_itens.Count > _maximo)
            {
                string maisAntigo = _itens.MinBy(p => p.Value.CriadoEmUtc).Key;
                _itens.Remove(maisAntigo);
                if (_ultimoId == maisAntigo) _ultimoId = null;
            }
        }
    }

    public ChatDiagnosticContextResponse? Obter(string? empresaId)
    {
        lock (_trava)
        {
            RemoverExpirados();

            string? alvo = string.IsNullOrWhiteSpace(empresaId) ? _ultimoId : empresaId;
            if (alvo == null) return null;

            return _itens.TryGetValue(alvo, out var entrada) ? entrada.Contexto : null;
        }
    }

    private void RemoverExpirados()
    {
        DateTime limite = DateTime.UtcNow - _validade;
        var expirados = _itens.Where(p => p.Value.CriadoEmUtc < limite).Select(p => p.Key).ToList();
        foreach (string id in expirados)
        {
            _itens.Remove(id);
            if (_ultimoId == id) _ultimoId = null;
        }
    }
}
