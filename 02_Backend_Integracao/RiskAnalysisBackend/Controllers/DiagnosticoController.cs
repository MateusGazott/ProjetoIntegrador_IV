using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc;
using RiskAnalysisBackend.Models;
using RiskAnalysisBackend.Services;

namespace RiskAnalysisBackend.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class DiagnosticoController : ControllerBase
{
    private static readonly Regex PadraoEmpresaId = new(@"^[A-Za-z0-9_\-]{1,64}$", RegexOptions.Compiled);

    private readonly IRiskPredictionService _servico;

    public DiagnosticoController(IRiskPredictionService servico) => _servico = servico;

    [HttpPost("analisar")]
    [ProducesResponseType(typeof(RiskDiagnosisResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErroResponse), StatusCodes.Status400BadRequest)]
    public ActionResult<RiskDiagnosisResponse> Analisar([FromBody] FinancialDataRequest request) =>
        Ok(_servico.AnalisarDadosFinanceiros(request));

    [HttpGet("chat-consulta")]
    [ProducesResponseType(typeof(ChatDiagnosticContextResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErroResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErroResponse), StatusCodes.Status404NotFound)]
    public ActionResult<ChatDiagnosticContextResponse> ConsultarParaChat([FromQuery] string? empresaId = null)
    {
        if (!string.IsNullOrWhiteSpace(empresaId) && !PadraoEmpresaId.IsMatch(empresaId))
            return BadRequest(new ErroResponse("empresaId inválido."));

        var contexto = _servico.ConsultarDiagnosticoParaChat(empresaId);
        if (contexto == null)
            return NotFound(new ErroResponse("Nenhum diagnóstico encontrado. Envie os dados financeiros primeiro em POST /api/diagnostico/analisar."));

        return Ok(contexto);
    }

    [HttpGet("atributos-modelo")]
    [ProducesResponseType(typeof(ModelMetadata), StatusCodes.Status200OK)]
    public ActionResult<ModelMetadata> ObterAtributos() => Ok(_servico.ObterMetadadosModelo());

    [HttpGet("saude")]
    [ProducesResponseType(typeof(ServiceStatusResponse), StatusCodes.Status200OK)]
    public ActionResult<ServiceStatusResponse> Saude() => Ok(_servico.ObterStatus());
}
