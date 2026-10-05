# Curriculum Parasite 60 HP с canonical test start

06.10.2026. Цель — получить опыт более длительного сопровождения и добивания, которого почти не было в HP20 width pilot. Архитектура остаётся MLP 386→64→64, actor 8/value 1; GRU/attention не смешиваются с изменением fixture.

Root: `workspace/artifacts/combat-ppo-curriculum-hp60-v1-20261006`. Start weights/checkpoint — mainline HP20 fourth update SHA256 `9920296b170acaa1f184f61ea900d6579af88b4999978ce1e76d9bbbd76cb01b`; более поздние width candidate checkpoints не выбирались по eval. Reward v2 и PPO config сохраняются, optimizer/RNG/consumed-rollout history resume без дополнительного reset. Старый опыт не используется повторно.

Замороженный протокол: четыре fresh batch × четыре инстанса, x2, отдельный seed каждого эпизода 15100–15115; 300 **post-barrier** игровых кадров; release game_frame 100, Blaster READY gunframe 9, post-frame seed reset. HP60 применяется один раз на release; никаких изменений здоровья/AI/aim во время боя. HP и native receipts не поступают в policy features. Full world reset не доказан: [диагностика повторений](learned_combat_reset_clock_v1.md) сохраняет отрицательный seed 14902.

Fixed fourth update, затем четыре deterministic eval до/после на обычных 175 HP, seeds 15200–15203. Eval не поступает в trainer. Показатели — включённые PPO kill reward windows, actual outgoing/received health damage, first-life deaths/kills и diagnostics. Legacy rules-specific harness acceptance не используется как learned-policy приёмка. Native provenance, phase effects, command dispatch, reward SHA и Go/PyTorch replay остаются обязательными.
