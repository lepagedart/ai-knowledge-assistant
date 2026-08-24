# Harbor & Hearth Café Demo Script

The fictional documents in `demo_documents/harbor_and_hearth/` demonstrate the
grounded-answer behavior. The expected results below describe evidence and
citations, not exact model wording.

## 1. Supported answer

Ask: “How far in advance should I call out for a shift?”

Expected behavior: answer from the call-out policy that a team member should
contact the shift lead at least two hours before the scheduled start, except for
an emergency as described in that same policy. Cite the call-out policy section.

## 2. Multi-document answer

Ask: “What should a team member do if a guest asks for a refund because of an
allergen concern?”

Expected behavior: combine the menu/product reference’s allergen escalation
instructions with the refund/service-recovery policy’s manager approval and
documentation process. Cite both documents and avoid inventing a medical or
refund guarantee.

## 3. Citation traceability

Ask: “What are the opening cash-drawer steps?”

Expected behavior: summarize the opening SOP’s documented cash-drawer count,
float verification, discrepancy escalation, and register sign-in steps. Cite the
opening/closing SOP section and show an inspectable supporting excerpt.

## 4. Unsupported question

Ask: “What is the CEO’s home address?”

Expected behavior: clearly state that the provided materials do not support an
answer. Do not guess, search outside the materials, or provide a citation.
