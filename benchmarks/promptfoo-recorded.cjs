// Local replay only. No network, model, fabricated citations, or judge calls.
module.exports = class RecordedAnswers {
  id() { return "recorded-contract-answers"; }
  async callApi(_prompt, context) {
    return { output: context.vars.recorded_answer };
  }
};
