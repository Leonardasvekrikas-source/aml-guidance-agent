# Judge agreement audit

The groundedness figure in the README comes from an LLM judge. This file is the
check on that judge.

**How to complete it.** For each case below, read the answer and the passages,
then write `agree` or `disagree` on the `Human:` line — `agree` meaning you
reach the same verdict the judge did, `disagree` meaning you do not. Add a
sentence saying why when you disagree.

Grade before looking at the judge's reasoning if you can. Reading its
justification first makes agreement more likely and the audit worth less.

Then run `make judge-audit-score`.

---


## Case 1 — `a09`

**Question.** How can an online shopping platform be used to shift value from one member of a terrorist cell to another abroad, when no bank transfer ever passes between the two of them?

**Answer given.**

> The corpus (principally FATF's Comprehensive Update on Terrorist Financing Risks) describes exactly this typology under \"E-commerce platforms and online marketplaces\" (EPOMs).\n\nThe core mechanism is that value moves as goods rather than as money. In the scheme FATF sets out, the first cell member buys items and ships them to his accomplice via an online marketplace; the accomplice resells them in the other jurisdiction and uses the sale proceeds to finance terrorism. No payment ever needs to pass between the two of them — the goods themselves are the carrier of value, and the disguise is the apparently ordinary consumer transaction. The money the second member ends up holding comes from an unconnected retail buyer.\n\nA second variant is that the two members act as counterparties to each other, posing as buyers and sellers through fraudulent or complicit shop fronts, and apply classic trade-based techniques — over- and under-invoicing, or multiple invoicing for the same goods — so that the imbalance between the true value of the goods and the price paid moves value in the desired direction. The Wolfsberg/ICC/BAFT trade finance principles explain why this works: it depends on collusion between buyer and seller, often because the same persons control both sides. FATF/Egmont's TBML report documents the same underlying \"commodity as currency\" principle in a terrorist financing case where goods were shipped and settled in cash by operatives of the same organisation at the far end.\n\nDetection therefore has to come from trade-style rather than payment-style red flags: inconsistent pricing, vague goods descriptions, multiple invoices for one shipment, third-party payment for goods, and an online presence inconsistent with the purported business.

**Passages.** `results/traces/f3bad3aeae03.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 2122-2124 describe the EPOM scheme (first actor purchases items, sends via EPOM to accomplice who sells and finances terrorism) and mention over/under invoicing on EPOMs, but nothing in the passages attributes the source of the money the second member holds as coming from 'an unconnected retail buyer,' nor do they cite a Wolfsberg/ICC/BAFT explanation tied to this EPOM scheme (that content is from a different, unrelated chunk 4557 about trade finance principles generally, not linked to EPOMs). The 'commodity as currency' case (chunk 208) is a separate case about cash payment for goods shipped from Country C to A, not clearly the same 'underlying principle' as the EPOM scheme as the answer implies.

Unsupported assertions named:
- The money the second member ends up holding comes from an unconnected retail buyer
- The Wolfsberg/ICC/BAFT trade finance principles' explanation of collusion is explicitly linked to explaining why the EPOM buyer/seller variant works
- The FATF/Egmont TBML case (commodity as currency, Box on Israel) documents 'the same underlying commodity as currency principle' as the EPOM scheme specifically

</details>

**Human:** 

**Note:** 

---

## Case 2 — `a15`

**Question.** A day-spa business account we monitor takes in nearly all of its money as cash, with most deposits posting late at night and far exceeding what a single storefront should earn, and it never sends any payroll or ACH payments. The listed phone number also appears on escort advertising sites. Which specific indicators does this combination match?

**Answer given.**

> This fact pattern maps most directly onto FinCEN's human trafficking "front company" typology (FIN-2020-A008), which names massage businesses and escort services as classic sex-trafficking fronts that commingle illicit proceeds with apparently legitimate revenue, plus the advisory's financial red flag that transactions are inconsistent with the customer's expected activity/line of business. The escort-ad linkage ties to the advisory's discussion of traffickers advertising commercial sex online and its red flag on transactions with services that host advertising content for illicit services. Treasury's 2024 NMLRA independently identifies illicit massage and escort services as high-risk human-trafficking sectors and describes cash-intensive front companies used to commingle illicit and legitimate proceeds. The corpus does not contain an indicator worded specifically as "no payroll or ACH," though the absence of disbursements to employees or payroll companies appears in the corpus as bank-record evidence that a business was a shell. Remember the advisory's own caution that no single red flag is determinative, and its SAR filing instructions.

**Passages.** `results/traces/4640ad751571.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

The answer weaves together several passages (FinCEN human trafficking advisory, NMLRA 2024, Seminerio shell company case) into a unified analytical narrative that draws connections and equivalences not explicitly stated in the passages themselves, particularly overstating links between payroll/ACH indicators and shell company bank-record evidence. While individual facts (massage/escort fronts, advertising red flags, NMLRA sector listings) are each supported by their respective passages, the synthesized claims about how these connect exceed what is directly stated or entailed by the text.

Unsupported assertions named:
- That NMLRA (2024) attributes 'cash-intensive front companies used to commingle illicit and legitimate proceeds' specifically to human trafficking sectors like illicit massage and escort services—the passages describe cash-intensive front companies (convenience stores, restaurants, liquor stores, auto shops) in the context of general drug trafficking money laundering, and separately list human trafficking industries (including illicit massage and escort services) without explicitly tying the front-company/commingling mechanism to those specific sectors in the same passage.
- The claim that 'no payroll or ACH' specifically appears as an indicator matching bank records showing no disbursements to employees/payroll companies—the passages show these as two separate, unrelated items (a general trade-based ML structural indicator about payroll transactions, and a specific Seminerio shell company case about no payroll disbursements), not as a synthesized/connected point the answer implies.
- The overall synthesis presenting these as a single coherent cross-referenced red-flag pattern ('maps most directly onto,' 'ties to,' 'independently identifies') overstates the direct textual connections between separately sourced passages that do not explicitly cross-reference each other in this manner.

</details>

**Human:** 

**Note:** 

---

## Case 3 — `a22`

**Question.** In trade finance, how do over invoicing, under invoicing and short shipping each transfer excess value, and which party to the trade gains in each case?

**Answer given.**

> I could not substantiate the rejected claim. Repeated searches using varied vocabulary ("short shipping", "seller ships fewer goods than invoiced", "over-/under-shipment", "misrepresentation of quantity", "Short Shipping / Over Shipping / Falsely Described Goods") returned no passage that defines "short shipping" or names the party that gains from it, so I am dropping that claim rather than restating it as a negative finding about the corpus (which no single passage can verify). What the corpus does support is set out below: the Wolfsberg/ICC/BAFT methods table defines Phantom Shipping and attributes a gaining party for over-invoicing, under-invoicing and multiple invoicing, and FATF/Egmont, the FATF professional money laundering report and the UK NRA describe phantom/ghost shipping and quantity- and quality-based misrepresentation more generally.

**Passages.** `results/traces/78f4dcf3f41e.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunk 4557 explicitly defines 'Short Shipping' and states its effect is similar to over-invoicing, directly contradicting the answer's claim that no passage defines short shipping or its beneficiary. Therefore the answer's assertion of an information gap is not supported by the passages provided.

Unsupported assertions named:
- The claim that no passage defines 'short shipping' or names the party that gains from it is inaccurate, since chunk 4557 explicitly defines 'Short Shipping' as the seller shipping less than the invoiced quantity/quality, misrepresenting value in a way similar to over-invoicing, which implies the buyer/payer effectively loses value while the seller gains excess value akin to over-invoicing.
- The answer asserts it is 'dropping' the claim rather than addressing it, but the passages do contain sufficient information (chunk 4557) to characterize short shipping and its effect, making the refusal to answer not fully grounded in what is available.

</details>

**Human:** 

**Note:** 

---

## Case 4 — `a25`

**Question.** Our screening only compares customer names against the UK sanctions list and it never produces a hit. What does the FCA say that approach can miss, what detection technique does it point firms towards, and where can we get warning indicators for that behaviour?

**Answer given.**

> The FCA's Financial Crime Guide (FCG 7.2.3A) says explicitly that simple screening of names against the Consolidated List may not always identify sanctions evasion involving third parties, and that alternative detection techniques may be needed. The technique it points to as good practice is data analytics to identify customers who may be close associates or dependents of designated persons, or who have transactional links with them, and so pose a higher risk of sanctions non-compliance. For warning indicators, the FCA directs firms to the red flags for sanctions evasion set out in alerts issued by the National Economic Crime Centre (NECC); it also signposts OFSI's website and FCTR 8/FCTR 15 for wider sanctions guidance.\n\nSeparately, your \"never produces a hit\" symptom maps onto an FCA poor-practice example: a firm that does not understand how its automated screening tool is calibrated and does not check whether the number of hits is unexpectedly high or low. Calibration that is not sensitive enough can mean name variations go undetected, so the FCA expects calibration tailored to the firm's risk plus effectiveness controls such as sample testing and tuning. Wider typology work also confirms that evasion commonly runs through intermediaries, obscured beneficial ownership, virtual assets and shipping — none of which a plain name match against a designated-persons list will catch.

**Passages.** `results/traces/79ef9dac7c62.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

The passages support the core claims about FCG 7.2.3A (screening limitations, data analytics as good practice, NECC alerts) and the calibration/poor-practice example, and separately support the APG typologies about intermediaries, BOI, virtual assets, and shipping. However, the answer improperly blends these two unrelated source documents into a single continuous argument, implying the FCA guidance and APG typology report are connected or mutually reinforcing on evasion detection, which the passages do not support.

Unsupported assertions named:
- The FCA directs firms to FCTR 8/FCTR 15 specifically as guidance on sanctions evasion warning indicators (FCTR is signposted for general sanctions topics like senior management responsibility, risk assessment, screening, dual-use goods in trade finance—not specifically tied to evasion detection technique in this way)
- Wider typology work (APG Yearly Typologies Report) confirms evasion commonly runs through virtual assets and shipping in a way connected/integrated with the FCA guidance discussion - this conflates two separate, unrelated documents (FCA's FCG on UK sanctions screening and APG's proliferation financing typologies report) as if they form one coherent narrative
- The claim that OFSI's website is signposted specifically as part of the discussion on evasion detection alternative techniques (it's listed under general 'sources of further information' in FCG 7.4, not tied specifically to evasion techniques)

</details>

**Human:** 

**Note:** 

---

## Case 5 — `a26`

**Question.** A bank we clear payments for keeps sending us instructions where the payer field holds filler text instead of a real name. What are we as the intermediary supposed to do when we spot that, and who ultimately has to make sure the payer's details are accurate?

**Answer given.**

> The corpus treats "filler"/meaningless payer fields as a payment-transparency failure that you, as the intermediary in the chain, are expected to detect, investigate and act on — but the primary duty to obtain and verify accurate payer details sits with the originating (ordering) institution that holds the payer as its customer.

What you must do as intermediary: monitor the respondent's payment messages specifically to detect ones lacking required originator/beneficiary information, including meaningless fields, consistent with FATF R.16 (Basel Committee). Treat message quality as part of ongoing monitoring and verify the reliability of the respondent's controls. Have documented risk-based policies and procedures for how such payments are handled (including in straight-through processing), send targeted requests for information (RFIs) to the respondent on the specific flagged payments, and track responses within a defined time limit. Never strip or truncate payer/payee data before passing the payment on — that is explicitly poor practice, and truncation undermines your own and every downstream bank's screening and monitoring. Feed the respondent's track record on payer data into the ongoing review of the correspondent relationship; persistent failure to respond or to provide complete payer information is itself a red flag. Escalation options run from refusing the transaction and filing a suspicious transaction report, through limiting or restricting services to give the respondent a chance to remediate, up to termination where the risk cannot be managed. An incoming transaction not accompanied by the required payer/payee information is a recognised risk-increasing factor under the EBA Risk Factors Guidelines.

Who owns accuracy: the ordering/originating institution. It is responsible for correct message format and for ensuring originator and beneficiary information accompanies the transfer, and it must obtain, hold and submit *accurate* (i.e. verified) originator information consistent with its own CDD. Institutions further down the chain are not required to verify the originator's name for accuracy, but must review it for STR monitoring and sanctions screening. Note two caveats: correspondents are not required to do CDD on the respondent's customers (that is the respondent's job), yet ultimate responsibility for implementing AML/CFT measures in your own institution remains with you.

**Passages.** `results/traces/4936f4d5e286.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Most individual claims trace to real passages (Basel/FSB correspondent banking guidance on monitoring for meaningless fields, RFI use, FCG examples on stripping payee info, EBA risk factors, and R.16 responsibilities), but the answer stitches together points from distinct frameworks (VASP travel rule, EBA guidelines, Basel correspondent banking, FCG) into a single unified obligation narrative not explicitly presented together, and some specifics (documented policies for meaningless fields, escalation ladder tied to this issue) are extrapolated beyond what is stated.

Unsupported assertions named:
- Have documented risk-based policies and procedures for how such payments are handled (including in straight-through processing) — passages mention 'risk-based policies and procedures for determining how to handle such transactions' and 'straight-through processing' but do not explicitly link a documentation requirement specifically to meaningless/filler fields as stated.
- Feed the respondent's track record on payer data into the ongoing review of the correspondent relationship — this is stated in the FCG passage as an example of good practice, not as a general 'must do' obligation across all frameworks as the answer implies.
- Escalation options run from refusing the transaction and filing an STR, through limiting/restricting services, up to termination where risk cannot be managed — this sequence is described in the correspondent banking guidance context generally, not specifically tied to meaningless/filler payer fields.
- The claim that institutions further down the chain 'are not required to verify the originator's name for accuracy, but must review it for STR monitoring and sanctions screening' is stated for beneficiary institutions in the VASP/travel rule context, not generically for 'intermediary' correspondent banking chains handling meaningless fields as the answer frames it.
- The specific combination and framing implying this is a unified cross-context rule (combining VASP travel-rule wording with correspondent banking guidance) is not explicitly stated together in any single passage, making the synthesized narrative only partially grounded.

</details>

**Human:** 

**Note:** 

---

## Case 6 — `a27`

**Question.** We have a group buying several high-value cars with cash and putting the registrations in the names of people outside the group, while the same principals run a number of used-jewellery outlets. What scheme does that combination point to?

**Answer given.**

> The combination you describe is the classic "high-value goods / cash-conversion" laundering model, with three interlocking components that the corpus treats as distinct but complementary typologies:

1. **Converting bulk criminal cash into high-value goods.** Cash is pooled and then integrated by buying high-value goods — the UK NRA lists cars, gold, jewellery and watches expressly as assets bought with criminal funds to act as a long-term store of value that can later be liquidated or "re-laundered". High value dealer (HVD) status bites at €10,000 cash, and structuring across multiple lower-value goods is a recognised evasion of it.

2. **Nominee / third-party registration to break the ownership trail.** Putting titles in the names of people outside the group is the "front man / surrogate" technique: a surrogate whose name attracts less attention is declared the owner, concealing the principal's connection to the asset and frustrating tracing and confiscation. There is a directly analogous US conviction for using a nominee to purchase automobiles to conceal the source of the funds.

3. **Used-jewellery outlets as cash-intensive front businesses and as a precious-metals/stones channel.** Front companies combine illicit proceeds with legitimate business revenue; second-hand/used-goods and precious metals dealing lets an OCG convert illicit cash into a stable, anonymous, easily exchangeable asset, and undeclared cash income can be used to buy undeclared stock.

Taken together, this reads as an organised-crime cash-placement and integration scheme running through unregulated or under-regulated high-value goods dealers, with nominee title-holding for asset protection — and, if the cars or gold are moved cross-border or resold through the outlets, it shades into trade-based money laundering, for which vehicle purchase with criminal cash plus resale chains and precious-metals dealers are documented FATF/Egmont patterns.

**Passages.** `results/traces/4237f351a6c3.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

While many individual facts (cash used to buy gold/jewellery/watches/cars as store of value, front men/nominee concealment of ownership, HVD cash-intensive business risks, Mudd nominee car case) are supported by specific passages, the answer synthesizes these into an integrated 'three interlocking components' typology and draws connections (structuring to evade thresholds, TBML shading) that are not explicitly stated in the passages and represent added interpretive claims.

Unsupported assertions named:
- Structuring across multiple lower-value goods as a recognised evasion of the €10,000 HVD cash threshold is not stated in the passages (chunk 5900-5901 discusses HVDs and multiple lower-value goods laundering but does not frame it as 'structuring... to evade' the threshold)
- The specific claim that there is a 'directly analogous US conviction for using a nominee to purchase automobiles to conceal the source of the funds' is supported (Stephen Mudd Jr. case), but the broader synthesis framing nominee use as a general 'front man/surrogate' asset-protection technique combined with front companies and TBML into one integrated 'organised-crime cash-placement scheme' is an interpretive construction not explicitly stated in the passages
- The claim that 'if the cars or gold are moved cross-border or resold through the outlets, it shades into trade-based money laundering' is an inference not directly supported by any single passage describing this exact combination
- The overall synthesis presenting these three typologies as 'interlocking components' of a single named model is analytical framing beyond what is stated in the source passages, which treat cash-intensive businesses, front men/nominees, and precious metals/HVD risks as separate topics in different documents

</details>

**Human:** 

**Note:** 

---

## Case 7 — `a30`

**Question.** Our relationship managers and compliance officers keep deadlocking over whether to take on high-risk corporate clients, and when the decision is escalated the revenue side usually prevails. How should that escalation body be structured so commercial interests cannot carry the decision?

**Answer given.**

> The corpus supports a governance model in which escalation and higher-risk relationship decisions are structured around the three lines of defence, with the AML/compliance function positioned independently of the revenue-generating business, and with decisions taken (and documented) at senior management level. Basel sets out the three lines and requires the chief AML/CFT officer to have stature, authority and a direct reporting line to senior management or the board; Wolfsberg requires the escalation path to be independent and available to all staff; StAR/EBA place termination-or-continuation and PEP approval decisions with senior management (or a committee including at least one senior manager), with the group AML/CFT compliance officer involved in higher-risk cases; and the FCA/EBA warn that profitability and relationship-manager closeness must not drive risk decisions. I could NOT support the claim that the audit committee or board should be involved in oversight of decisions to terminate or continue higher-risk relationships — that recommendation in the StAR text is directed at a different (narrower) role, so I have dropped it and substituted what the corpus actually says about board/audit-committee involvement.

**Passages.** `results/traces/b5ab7b817e82.json`

**Judge verdict.** NOT GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

The passages support the three-lines-of-defence framework, independence of compliance, and senior management approval/documentation requirements, but the StAR Bank material cited is specifically about PEP customers, not higher-risk relationships generally as the answer implies. The answer's self-correcting final paragraph also mischaracterizes what the StAR passages say about audit committee/board involvement, which is tied to the PEP list, not general termination-or-continuation decisions.

Unsupported assertions named:
- The claim that StAR/EBA place termination-or-continuation decisions with senior management applies broadly to 'higher-risk relationships' generally—passages 7975-7976 and 7970-7972 specifically discuss this only in the context of PEP customers, not higher-risk relationships in general.
- The claim that the group AML/CFT compliance officer's involvement in higher-risk cases is a general rule for termination/continuation decisions—passage 7971 specifically limits this recommendation to PEP approval decisions, not all higher-risk relationship decisions.
- The final paragraph's claim that StAR recommends board/audit-committee involvement for termination-or-continuation of higher-risk relationships (even after being 'corrected')—the passages (7975-7976) state audit committee/board review applies specifically to the PEP list annually, not to termination-or-continuation decisions for higher-risk relationships broadly.
- The answer frames the entire discussion as being about 'higher-risk relationships' generally, but most of the StAR passages (7970-7976, 7988-7989) are specifically and narrowly about PEP customers, which is a narrower scope than presented.

</details>

**Human:** 

**Note:** 

---

## Case 8 — `a01`

**Question.** A trading company's account keeps receiving large incoming wires that are broken up and pushed out the same day to accounts with no obvious commercial link, so the balance sits near zero overnight. Which trade-based laundering indicator does that pattern correspond to?

**Answer given.**

> The pattern you describe maps directly onto the FATF/Egmont TBML risk indicator for a "pay-through" (or "transit") account: an account of a trade entity showing rapid movement of high-volume transactions with a small end-of-day balance and no clear business rationale. The specific sub-indicator listed under it is almost a verbatim match — incoming wire transfers to a trade-related account that are split and forwarded onward to multiple non-related accounts with little or no connection to commercial activity. Note the caveat FATF attaches to its list: these indicators are not conclusive, are not exclusive to TBML, and may point to other forms of money laundering or illicit activity, so the pattern warrants further analysis rather than being treated as determinative on its own.

**Passages.** `results/traces/e6ab12086a34.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 20-21 verbatim describe the 'pay-through'/'transit' account indicator with rapid movement of high-volume transactions and small end-of-day balance, and the sub-indicator about split incoming wire transfers forwarded to non-related multiple accounts with little or no connection to commercial activity. Chunk 16 supports the caveat that these indicators are not conclusive and may relate to other forms of ML or illicit activity, not exclusively TBML.

</details>

**Human:** 

**Note:** 

---

## Case 9 — `a03`

**Question.** Apart from checking price and quantity discrepancies, what practical checks on the paperwork a bank receives to pay a trade transaction have actually helped identify TBML?

**Answer given.**

> The Wolfsberg/ICC/BAFT Trade Finance Principles summary-of-controls table for SBLCs and guarantees separates the claim-presentation stage from the payment stage. At presentation of claim documents (whether by Party Y directly or via Bank B to Bank A), the review is for new principal parties or countries not mentioned in the SBLC/Guarantee, plus sanctions list screening and AML checks. At the payment stage, it is the names on the payment instruction that are reviewed against sanctions lists and AML checks. Related pre/post-event risk indicators for claims and payment instructions include last-minute changes to payment instructions, claims made shortly after issuance, continuous claims and claim pressure tactics.

**Passages.** `results/traces/382224188030.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 4627-4628 detail the summary of controls table showing presentation of claim documents reviewed for new principal parties/countries not mentioned in SBLC/Guarantee plus sanctions and AML checks, and payment stages reviewed against names on payment instructions via sanctions/AML checks. Chunk 4627 also lists claims and payment instruction risk indicators including last-minute changes, claims made shortly after issuance, continuous claims, and claim pressure tactics, matching the answer exactly.

</details>

**Human:** 

**Note:** 

---

## Case 10 — `a04`

**Question.** When does a customer's use of a virtual asset ATM or kiosk actually amount to a red flag, and when is it not enough on its own?

**Answer given.**

> FATF's 2020 report "Virtual Assets – Red Flag Indicators of Money Laundering and Terrorist Financing" includes an indicator for use of VA ATMs/kiosks: use despite higher transaction fees (including machines commonly used by mules or scam victims), or use in high-risk locations where increased criminal activity occurs. FATF is explicit that a single use of an ATM/kiosk is not by itself a red flag — it becomes one when combined with a high-risk location, repeated small transactions, or other additional factors. The indicator appears alongside indicators about VASPs with demonstrably weak or non-existent KYC, and the report more generally warns that indicators should not be viewed in isolation and should not be the sole determinant of whether to file an STR. Separately, FATF lists customer-profile indicators — purchases of large amounts of VA not substantiated by available wealth or inconsistent with the customer's historical financial profile, and customers significantly older than the average platform user conducting large numbers of transactions — which may point to money laundering, money mule activity, scam victimisation or elder financial exploitation.

**Passages.** `results/traces/8aba0452c614.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

All claims—the ATM/kiosk indicator with its two bullet conditions, the caveat that a single ATM/kiosk use is not sufficient alone, the general warnings about not viewing indicators in isolation and not solely determining STR filing, the weak/non-existent KYC VASP indicator, and the customer-profile indicators about large VA purchases inconsistent with wealth/profile and older customers engaging in high transaction volumes—are directly stated in chunks 940-941, 933-934, and 944-945.

</details>

**Human:** 

**Note:** 

---

## Case 11 — `a05`

**Question.** When a crypto exchange sends a customer's coins to another regulated exchange, which details about the sending customer must the exchange itself have verified, and which details about the receiving party can it pass on without verifying?

**Answer given.**

> Under the FATF "travel rule" (R.16 as applied to virtual assets via INR.15 para 7(b)), the sending (ordering) VASP must obtain, hold and transmit both originator and beneficiary information — but only the originator data has to be accurate/verified. So the exchange must have verified its own customer's full name, and hold that customer's wallet/account number and one of address / national ID number / customer ID number / date and place of birth. The beneficiary details it passes on — the receiving party's name and beneficiary account (wallet) number — are required to be submitted but need not be verified for accuracy by the sender, though the sender must still sanction-screen the beneficiary name and monitor for suspicion. Correspondingly, the receiving exchange may assume the originator data was verified upstream and does not have to verify it, but must have verified its own customer (the beneficiary) and check the incoming data matches.

**Passages.** `results/traces/9d89d82d781b.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Passages (chunks 723-726 and Table 1) confirm that ordering VASPs must obtain and hold required, accurate originator information (name, account/wallet number, and address/national ID/customer ID/date-place of birth) and required (but not necessarily accurate) beneficiary information, while sanction-screening the beneficiary and monitoring for suspicion. They also confirm beneficiary VASPs need not verify originator data (may assume it was verified by ordering VASP) but must verify their own customer's (beneficiary's) name/account for accuracy and confirm consistency with received data.

</details>

**Human:** 

**Note:** 

---

## Case 12 — `a06`

**Question.** A money-remittance business banks with us and runs everything through a single account - its own operating costs and the payments it pushes through for the people who use its service. Should we make it split those flows into separate accounts, and are we obliged to identify its end users ourselves?

**Answer given.**

> On the account structure: nothing in the corpus makes segregation mandatory, but FATF explicitly suggests it as a risk-management option — a bank "could consider encouraging or requiring" its MVTS customers to run one account for their own corporate/settlement activity and a separate account for the payments they process on behalf of their own customers, precisely because the two activities carry different risks and separation makes monitoring more effective. Separate accounts are also suggested for higher-risk activity and for nested/downstream relationships. So requiring the split is a defensible, guidance-supported control, framed as good practice rather than an obligation.

On identifying the end users: no. FATF, the Basel Committee and the EBA all state that you are not required to conduct CDD on your customer's customers ("KYCC"). Your customer is the remitter; the remitter is responsible for CDD on its own users. Your obligations are to do proper (and, where risk warrants, enhanced) due diligence on the remitter itself, to understand its business, target markets and customer segments, to monitor the account, and to document in writing how the account may be used and what third-party usage is permitted. Two qualifications: (a) if the arrangement is effectively a payable-through account, where the remitter's customers transact directly on the account, you must be able to identify the ultimate user and satisfy yourself the remitter has done sufficient CDD and can hand over individual CDD records on request; and (b) when monitoring flags a specific transaction, you can make a targeted request for information about the underlying customer without that amounting to a CDD obligation. Finally, guidance warns against blanket de-risking — restrict or limit services before terminating.

**Passages.** `results/traces/3a6a90d6b216.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

All claims are directly supported: chunks 2393-2394 describe encouraging/requiring separate accounts for MVTS corporate vs. correspondent activity and separate accounts for higher-risk/nested relationships; chunks 2392-2394 and 2384/2377-2378 confirm no CDD obligation on customer's customers (KYCC), the payable-through-account exception requiring identification of ultimate user, targeted RFIs not amounting to CDD, written agreements documenting responsibilities/third-party usage, and the de-risking guidance favoring limitation over termination (chunks 2390-2392).

</details>

**Human:** 

**Note:** 

---

## Case 13 — `a07`

**Question.** What factors influence the commission rate a professional money launderer charges a criminal client, and in what forms do PMLs actually take that commission?

**Answer given.**

> The FATF's Professional Money Laundering report is the corpus source on this. It says a set of overlapping factors set the fee/commission: the complexity of the scheme, the methods used, the launderer's knowledge of the predicate offence, and the level of risk the PML assumes — with rates further influenced by the countries/regions involved, the PML's reputation, the total amount laundered, the denomination of banknotes in cash cases, and how quickly the client wants the funds moved or concealed (tighter deadlines mean a higher commission). On the form of payment, the corpus is thinner: the defining feature of a PML is providing laundering services "in exchange for a commission, fee or other type of profit", and PMLs who are embedded in or close to an organised crime group may instead take a portion of the group's proceeds rather than a discrete fee. PMLs record commissions received in their own coded internal accounting systems. The corpus does not set out a fuller itemised list of payment forms (e.g. payment in kind).

**Passages.** `results/traces/230d7c2d8024.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunk 1128 lists complexity of scheme, methods used, knowledge of predicate offence, level of risk assumed, and factors like reputation, total funds laundered, denomination of banknotes, time requested (shorter time = higher commission), and new regulation/enforcement activity — all matching the answer. Chunks 1127 and 1132 support the definition of PML via commission/fee/profit and that those close to OCGs may receive a portion of proceeds rather than a fee, and chunk 1128 supports shadow accountancy tracking commissions received; the answer's caveat about the corpus being thinner on payment forms is also consistent with the passages not listing further detail.

</details>

**Human:** 

**Note:** 

---

## Case 14 — `a10`

**Question.** What signs in a trade finance file suggest the two counterparties to the deal are not genuinely independent of one another?

**Answer given.**

> The FATF/Egmont "Trade-Based Money Laundering: Risk Indicators" paper sets out a group of "structural risk indicators" relating to how a trade entity is constituted and registered. These cover unusually complex/illogical corporate structures (e.g. shell companies or companies in high-risk jurisdictions), registration or offices in jurisdictions with weak AML/CFT compliance, registration at likely mass-registration addresses, and business activity inconsistent with the stated address. A footnote to the mass-registration indicator adds that such addresses may include that of a trust and company service provider managing a number of shell companies for its customers. Further structural indicators concern staffing levels inconsistent with trading volume, names copying well-known corporations, unexplained dormancy and non-compliance with routine business obligations such as VAT filing. The paper stresses that a single indicator does not alone warrant suspicion.

**Passages.** `results/traces/2a1a733f9832.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 18-19 list the exact structural risk indicators cited (complex corporate structures, weak AML/CFT jurisdiction, mass-registration addresses with the trust/company service provider footnote, inconsistent business address, minimal staff vs. trading volume, name mimicking known corporations, unexplained dormancy, non-compliance with VAT filing), and chunk 17 states a single indicator does not alone warrant suspicion. All assertions in the answer are directly supported by these passages.

</details>

**Human:** 

**Note:** 

---

## Case 15 — `a11`

**Question.** On a life insurance policy, which premium payment patterns and which beneficiary changes should prompt a closer look?

**Answer given.**

> The corpus (EBA ML/TF Risk Factors Guidelines, sectoral guideline 14 on life insurance) explicitly lists product features that may *reduce* risk, and separately lists factors that may *increase* risk. I have kept these as separate, directly-sourced statements rather than inferring that the converse of each risk-reducing feature is a risk-increasing one, since the guidelines do not state that. Where increased risk from third-party payments, overpayments, cash and early termination is asserted, it is cited to passages that actually say so (Title I product risk factors, the customer-behaviour risk factor list, and the UK NRA).

**Passages.** `results/traces/94837d889d69.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

The passages (chunks 2727-2732) do list life insurance product risk-increasing factors (third-party payments, overpayments, cash payments, early surrender/termination) and separately list risk-reducing factors, matching the answer's description of the source material's structure and content.

</details>

**Human:** 

**Note:** 

---

## Case 16 — `a12`

**Question.** Where a firm uses an unattended remote customer onboarding solution in which the customer does not interact with an employee, what must the verification process include, and what should happen if the evidence captured is of insufficient quality?

**Answer given.**

> The EBA Guidelines on the use of Remote Customer Onboarding Solutions (EBA/GL/2022/15) set specific requirements for "unattended" solutions — i.e. where the customer does not interact with an employee to perform the verification. In those cases the firm must ensure any photograph or video is taken under adequate lighting and with sufficient clarity to allow proper verification of identity, that it is captured at the time the customer is actually performing the verification process, and that liveness detection is performed (the EBA deliberately made liveness detection mandatory for unattended/fully automated verification, though the firm may choose active or passive methods). More generally, images, video, sound and data must be captured in a readable format and with sufficient quality that the customer is unambiguously recognisable, and matching against the biometric data on the identity document must use strong and reliable algorithms, with additional controls where the required level of confidence is not reached. Where the evidence captured is of insufficient quality — creating ambiguity or uncertainty that affects the performance of the remote checks — the individual onboarding process should be interrupted and restarted, or redirected to face-to-face verification; the identification process should also not continue if technical shortcomings or unexpected connection interruptions are detected.

**Passages.** `results/traces/9ddd0155f27c.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

All claims—unattended solution requirements (lighting/clarity, timing, liveness detection), readable/quality capture of images/video/sound/data, strong algorithms for biometric matching with additional controls, interruption/restart or redirection to face-to-face on insufficient quality, and stopping on technical shortcomings/connection interruptions—are directly stated in chunks 2914, 2917-2918, and 2926-2927/2941. The chunks also confirm liveness detection was made mandatory specifically for unattended situations and that active/passive choice is left to the institution.

</details>

**Human:** 

**Note:** 

---

## Case 17 — `a13`

**Question.** Which virtual IBAN (vIBAN) arrangements do EU competent authorities rate as very significant risk, and what mitigating measure does the AML Regulation introduce for the lack of vIBAN transparency?

**Answer given.**

> Across the EBA's 2025 ML/TF risk Opinion, competent authorities (CAs) rate vIBAN risk as high for both payment institutions (PIs) and credit institutions (CIs), with the nature of the concern differing by sector: PI supervisors emphasise obscured account-holder identity, CI supervisors emphasise difficulty distinguishing virtual from traditional current account payments. The one arrangement CAs single out as a shared concern across both sectors is "cascading" vIBANs. The retrieved passage names cascading vIBANs as that common concern; I do not cite a specific severity label for them, as the retrieved text of the passage ends at the definition. Broader corpus material corroborates the underlying vIBAN vulnerability: lack of visibility over end users and reduced AML controls by intermediary PSPs, and a UK case in which vIBANs issued to ~60,000 companies enabled large-scale laundering.

**Passages.** `results/traces/efba9e671f37.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunk 3117 confirms CAs rate vIBAN risk high for PIs and CIs, with PI concerns about obscured account-holder identity and CI concerns about distinguishing virtual/traditional payments, and identifies cascading vIBANs as a shared 'very significant' risk for both sectors. Chunks 5882-5884 support the UK vulnerability description (lack of visibility, reduced AML controls, and the ~60,000-company vIBAN case with the OCG/ABP fraud).

</details>

**Human:** 

**Note:** 

---

## Case 18 — `a14`

**Question.** A corporate client with no prior history of buying digital currency contacts us in a hurry to purchase a large amount of Bitcoin, and in conversation shows they barely understand how it works. What does FinCEN say that combination is likely to mean?

**Answer given.**

> The behaviour you describe — a customer who displays little or no familiarity with convertible virtual currency (CVC) yet suddenly asks to buy it, especially in a large amount or in a hurry — is an explicit financial red flag indicator of ransomware in FinCEN's Advisory on Ransomware and the Use of the Financial System to Facilitate Ransom Payments (FIN-2021-A004, 8 November 2021), which updated and replaced the October 2020 advisory. FinCEN's stated interpretation is that this pattern may mean the customer is a ransomware victim being pressured into paying a ransom. Corroborating indicators in the same list include a customer with no or limited CVC transaction history sending a large CVC transaction outside normal business practice, a CVC address linked to known ransomware variants or payments, and irregular transactions between a high-risk-sector organisation and a DFIR or cyber insurance company known to facilitate ransomware payments. If you form a suspicion, ransomware-related suspicious transactions are treated as "situations involving violations that require immediate attention": call FinCEN's hotline and then file a SAR referencing the key term "CYBER FIN-2021-A004" in field 2 and the narrative, selecting SAR field 42 (Cyber Event). FinCEN also directs that the advisory be distributed to a specified list of roles including AML/BSA departments, cyber and security departments, customer service agents and bank tellers.

**Passages.** `results/traces/e8dcbb8ded62.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

The passages (chunks 3408-3423) directly support the advisory's title, date, purpose, the specific red flag about limited CVC knowledge yet purchasing CVC as indicating a possible ransomware victim, the corroborating red flags listed, the 'immediate attention' SAR characterization with hotline instructions, the SAR filing key term and field 42 instructions, and the distribution list including AML/BSA, cyber/security departments, customer service agents, and bank tellers.

</details>

**Human:** 

**Note:** 

---

## Case 19 — `a17`

**Question.** A Mexican importer client is wiring money to a Chinese chemical manufacturer, and the payment instructions and supporting invoice identify the substance only by a numeric registry identifier rather than naming it. Why do BSA filers treat that as an indicator of illicit precursor procurement?

**Answer given.**

> FinCEN's 2024 Financial Trend Analysis on fentanyl-related illicit finance is the corpus source on CAS numbers as a detection signal. It records that BSA filers often noted CAS numbers associated with fentanyl precursors in open-source advertisements, and that in some instances filers reported CAS numbers for fentanyl precursors referenced in payment instructions or in client-provided invoices. FinCEN explains the evasion rationale: CAS numbers are unique numerical identifiers for chemical substances, and criminals may use them to evade scrutiny by omitting explicit references to fentanyl. Separately — not stated as a combined pattern — filers identified chemical companies via e-commerce advertisements, product listings on company websites, and pointed language guaranteeing customs clearance. A footnote in the same document notes that payment instructions such as wire transfer memos often reference purposes of payment and/or invoice numbers for legitimate commercial activity; the document does not draw an explicit contrast between that baseline and a bare CAS number. The document does, however, separately flag generic payment instructions ("goods", "services") as something filers cited.

**Passages.** `results/traces/620826de8ebe.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 4290-4292 support all claims: CAS numbers in open-source ads, CAS numbers in payment instructions/invoices, the explanation that criminals use CAS numbers to evade detection by omitting explicit fentanyl references, and e-commerce/website/customs-clearance language identifying chemical companies. Chunk 4288 supports the note about generic payment instructions ('goods'/'services'), and chunk 4289 supports the footnote about payment memos referencing invoice numbers for legitimate activity.

</details>

**Human:** 

**Note:** 

---

## Case 20 — `a18`

**Question.** Why is a money order seller usually unable to screen the payee's name, and at what currency amount is it actually required to obtain and verify the purchaser's identity?

**Answer given.**

> Per Treasury's 2024 National Money Laundering Risk Assessment, there is no explicit BSA requirement for money order sellers to collect payee information, and the issuer does not learn the payee's name until the payee negotiates the instrument and it clears the banking system — so the seller typically cannot screen the payee at the point of sale. The BSA recordkeeping obligation for money orders requires the issuer/seller to obtain, verify and record customer identification only for currency purchases of money orders totaling $3,000 or more, which is a threshold criminals structure beneath.

**Passages.** `results/traces/9d54e4a302b6.json`

**Judge verdict.** GROUNDED

<details><summary>Judge reasoning (read after grading)</summary>

Chunks 4012-4014 directly state the $3,000 recordkeeping threshold, the lack of explicit payee-information requirement, that issuers don't know payee name until negotiation/clearing, and cite case examples of structuring below the threshold.

</details>

**Human:** 

**Note:** 

---
