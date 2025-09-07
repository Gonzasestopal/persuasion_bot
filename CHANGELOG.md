## v1.7.0 (2025-09-07)

### Feat

- topic checker singleton
- check topic anthropic adapter
- topic coherence rules

### Refactor

- update tests to work

## v1.6.0 (2025-09-07)

### Feat

- recommend user for stronger arguments more than 8 words

## v1.5.0 (2025-09-07)

### Feat

- restrict stance language and topic
- pass topic to context

### Fix

- set debate statelanguage after bot reply

### Refactor

- split integration tests

## v1.4.1 (2025-09-05)

### Refactor

- debate store (#7)

## v1.4.0 (2025-09-05)

### Feat

- add requirements

### Fix

- make test pass
- deps
- timeout

### Refactor

- use stance in domain

## v1.3.0 (2025-09-05)

### Feat

- nli performance
- add requirements
- nli port and dependency inyection
- **debate_store**: ports tests and adapter
- **concession**: split hypotesis into full sentences
- **concession**: shared debate state and locked language
- **concession**: use bidirectional scores
- **nli**: bidirectional scores
- **parser**: verify topic maximum length

### Fix

- timeout
- deps
- make test pass

### Refactor

- **infra**: split factories into infra
- **concession**: split service
- **concession**: use state store
- **debate store**: main service

## v1.2.3 (2025-09-05)

### Fix

- commitizen (#6)

## v1.2.2 (2025-09-04)

### Fix

- tag name (#4)

## v1.2.1 (2025-09-04)

### Fix

- release (#3)

## v1.2.0 (2025-09-04)

### Feat

- redis (#2)

## v1.1.0 (2025-09-03)

### Feat

- workflows
- **concession**: include thesis in nli
- working concession service
- **nli**: add nli provider
- **concession**: add service
- **errors**: wrap config errors as domain errors
- **factories**: use fallback llm as service provider
- **fallback**: fallback llm to handle requests
- **factories**: add anthropic as available provider
- **anthropic**: add adapter
- **llm**: improve system prompt
- **llm**: consider settings difficulty on new adapter
- **llm**: add medium difficulty system prompt
- **llm**: add win condition to avoid debating further
- **service**: insert full message history for prompt context
- **repo**: add all messages query
- **llm**: allow infinite message history if none is provided
- **llm**: implements debate openaiadapter to process message history
- **llm**: adapter generate conversation ready
- **api**: config errors translations
- **llm**: verify llm config before calling adapter
- **llm**: openai adapter
- **service**: use debate for continuing conversation
- **llm**: dummy debate method
- **messages**: dummy bot reply generator ready
- **llm**: inject env provider to factory
- **llm**: openai factory
- **repo**: ensure test user bot order messages is preserved
- **models**: replace dict data representation in favor of domain models
- **llm**: replies from dummy adapter
- **service**: llm dependency and reply
- **llm**: define adapter interface to satisfy tests
- **conversations**: update message service to check expired conversation
- **conversations**: persist messages
- **repo**: wire Postgres pool and repo into service
- **parser**: message side topic parsing
- **api**: add /messages route

### Fix

- **pg**: order messages from latest to oldest
- **factories**: limit of messages as pairs
- **deps**: use psycopg binary

### Refactor

- **concession**: update win text
- **policy**: typo naming
- **errors**: standarize naming
- **parser**: domain errors
- **errors**: wrap api with exception handler
- **errors**: wrap api errors
- **readme**: keep old errors section
- **tests**: remove unused memory fake
- messages route name into routes
- **migrations**: add migrate script that works for every environment
- **tests**: make adapter specs pass with default difficulty
- **dtos**: return api specified fields using dtos
- **adapter**: no longer check for history limit
- **service**: include full message history for prompt
- **openai**: reuse adapter methods
- **llm**: move creation specs into factory
- **llm**: update dummy adapter signature
- **repository**: split into adapter port
- **factories**: get llm and get service factories
- **llm**: adapter ports folder structure
