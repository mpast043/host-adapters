# POC Checklist - Governance Runtime

## Environment

- [ ] Python and dependencies installed
- [ ] CGF server starts and health endpoint responds
- [ ] Target host adapter configured

## Governance

- [ ] Adapter registration succeeds
- [ ] Denylisted action blocked
- [ ] Read-only action allowed
- [ ] Fail-mode behavior verified with CGF unavailable

## Evidence

- [ ] Event logs generated
- [ ] Replaypack generated
- [ ] Schema lint passes
- [ ] Contract suite passes

## Workflow Contract

- [ ] RUN directory created with required structure
- [ ] Selection artifacts generated in required path
- [ ] VERDICT and summary produced
- [ ] Retention archive created and copied

## Completion

- [ ] Customer reproduces the run using documented commands
- [ ] Customer signs off on governance and replay evidence
