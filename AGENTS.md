# Project Instructions

This directory contains the Retailrocket e-commerce personalized recommendation system project.

Before planning, analyzing data, designing models, writing code, or changing the interface, read `PROJECT_FRAMEWORK.md` and treat it as the canonical project scope.

## Durable Decisions

- The project studies whether browsing, cart, and purchase history can predict a visitor's next purchased item.
- Use the Retailrocket dataset and preserve its anonymity. Do not invent product names, images, prices, brands, or meanings for hashed item properties.
- The core storyline is: behavior analysis -> baselines -> personalized retrieval -> optional ML ranking -> time-based offline evaluation -> strategy analysis -> Chinese interactive UI.
- Use time-based splits and point-in-time features. Never allow future events or future item-property values to leak into training or recommendation inputs.
- Always compare personalized models against simple popularity baselines.
- Evaluate ranking quality and recommendation distribution, including Recall@K, NDCG@K or MRR@K, catalog coverage, long-tail exposure, diversity, and segmented performance.
- The first complete version should prioritize a defensible end-to-end workflow over adding many complex models.
- Treat Item-CF or ALS as the first personalized retrieval model. LightGBM ranking is the main advanced extension. GRU4Rec or other deep sequential models are optional.
- For item properties, use `categoryid` and `available` first. Hashed properties are optional content features and must not be assigned invented business meanings.
- The final interface should support selecting an anonymous visitor and historical session, inspecting prior behavior, switching models, viewing Top-N recommendations and reasons, and comparing recommendations with later observed purchases.
- Public offline data cannot prove online conversion, GMV, profit, or ROI improvement. State conclusions as offline evaluation results.

If a requested change conflicts with `PROJECT_FRAMEWORK.md`, surface the conflict before changing the project direction.
