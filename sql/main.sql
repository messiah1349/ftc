with portfolio_eff as (
select 
	p.*
	,isnull(dateadd(second, -1, lead(portfolio_dttm) over (PARTITION by loan_key order by portfolio_dttm)), Cast('1/1/2100' as datetime)
				) as effective_to
from 
	portfolio p
)
, overdue as (
select 
	pe.loan_key
	,max(datediff(day, b.date, portfolio_dttm)) - 1 as days_in_overdue 
from 
	portfolio_eff pe
left join 
	billing_plan b 
	on 
		pe.loan_key = b.loan_key
where 
	is_overdue = 1
group by 
	pe.loan_key
)
, default_matured as (
select 
	pe.loan_key
	,max(datediff(day, b.date, convert(date,getdate()))) - 1 as days_after_billing 
from 
	portfolio_eff pe
left join 
	billing_plan b 
	on 
		pe.loan_key = b.loan_key
group by 
	pe.loan_key
)
, default_status as (
select 
	dm.loan_key
	,dm.days_after_billing
	,o.days_in_overdue
from
	default_matured dm 
left join 
	overdue o 
	on 
		dm.loan_key = o.loan_key
)  
, previous_loans_overdue as (
select 
	l_current.loan_key, l_previous.loan_key as loan_key_previous
	,max(datediff(day, bp.[date], 
			case when pe.effective_to > l_current.issue_date then l_current.issue_date else pe.effective_to end)) as max_overdue_for_prev_loans
from 
	loan l_current
join 
	loan l_previous
	on 1=1
		and l_current.client_key = l_previous.client_key
		and l_current.issue_date > l_previous.issue_date
join 
	billing_plan bp 
	on 
		l_previous.loan_key = bp.loan_key
join 
	portfolio_eff pe 
	on 1=1 
		and pe.loan_key = l_previous.loan_key
		and pe.portfolio_dttm < l_current.issue_date
where 
	pe.is_overdue = 1
group by 
	l_current.loan_key, l_previous.loan_key
)
, previous_default as (
select 
	loan_key
	,count(*) as cnt_prev_overdue_loans
	,sum(case when max_overdue_for_prev_loans > 3 then 1 else 0 end) as prev_def3_cnt
	,sum(case when max_overdue_for_prev_loans > 10 then 1 else 0 end) as prev_def10_cnt
	,sum(case when max_overdue_for_prev_loans > 30 then 1 else 0 end) as prev_def30_cnt
from 
	previous_loans_overdue
group by 
	loan_key
)
, previous_loans as (
select 
	a.application_key
	,count(l.loan_key) as cnt_prev_loans
	,max(datediff(day, l.issue_date, a.application_dttm)) as oldest_loan_daydiff
	,min(datediff(day, l.issue_date, a.application_dttm)) as earlyest_loan_daydiff
	,AVG(l.amount) as avg_loan_amount
	,MAX(l.amount) as max_loan_amount
	,min(l.amount) as min_loan_amount
from 
	application a 
join 
	loan l 
	on 1=1
		and a.client_key = l.client_key
		and convert(date, a.application_dttm) > l.issue_date
group  by 
	a.application_key
)
, previous_payms as (
select 
	a.application_key
	,count(payms.amount) as cnt_prev_payms
	,sum(payms.amount) as sum_prev_amount
	,min(datediff(day, payms.paym_date, a.application_dttm)) as prev_paym_daydiff
from 
	application a
left join
	(
		select 
			p.amount
			,p.[date] as paym_date
			,l.client_key
			,l.loan_key
		from 
			loan l 
		join
			payment p 
			on 1=1
	 			and l.loan_key = p.loan_key
	) payms 
	on 1=1
		and a.client_key = payms.client_key
		and a.application_dttm > payms.paym_date
group by 
	a.application_key
)
select 
	cl.client_key
	,cl.fact_address
	,cl.inn
	,cl.legal_address
	,cl.legal_registration_date
	,cl.ogrn
	,cl.okved
	,cl.originator_registration_date
	,cl.organizational_form_key
	,app.application_dttm
	,app.application_key
	,app.rate_exp
	,app.amount as application_amount
	,app.status_id as application_status_id
	,app.term
	,org.name as originator_name
	,pr.name as product_name
	,moi.amount_executed_for_24_month
	,moi.arbitration
	,moi.BKI_flg_30
	,moi.BKI_flg_90
	,moi.[BKI_flg_90+] as bki_flg_90plus
	,moi.BKI_volume
	,moi.executive_production
	,moi.number_current_contracts
	,moi.number_current_with_similar_sum
	,moi.number_current_with_the_similar_work
	,moi.number_investors
	,moi.number_of_executed_for_24_month
	,moi.[number_of_the_following users] as number_of_the_following_users
	,moi.[number_with similar_sum_for_24_months] as number_with_similar_sum_for_24_months
	,moi.number_with_customer_for_24_months
	,moi.number_with_similar_work_for_24_months
	,moi.total_current_contracts
	,ds.days_after_billing
	,ds.days_in_overdue
	,pd.cnt_prev_overdue_loans
	,pd.prev_def3_cnt
	,pd.prev_def10_cnt
	,pd.prev_def30_cnt
	,pl.cnt_prev_loans
	,pl.oldest_loan_daydiff
	,pl.earlyest_loan_daydiff
	,pl.avg_loan_amount
	,pl.max_loan_amount
	,pl.min_loan_amount
	,pp.cnt_prev_payms
	,pp.sum_prev_amount
	,pp.prev_paym_daydiff
from 
	dwh.dbo.application app
join 
	dwh.dbo.loan l 
	on 
		app.application_key = l.application_key
join 
	dwh.dbo.client cl 
	on 
		app.client_key = cl.client_key
left join 
	dwh.dictionary.originator org 
	on 
	 	app.originator_key = org.originator_key
left join 
	dictionary.product pr 
	on 
		pr.product_key = app.product_key
left join 
	ai.modul_other_info moi 
	on 
		app.application_key = moi.application_key
left join 
	default_status ds 
	on 
		l.loan_key = ds.loan_key
left join 
	previous_default pd 
	on 
		l.loan_key = pd.loan_key
left join 
	previous_loans pl 
	on 
		pl.application_key = app.application_key
left join 
	previous_payms pp 
	on 
		pp.application_key = app.application_key
		
		
		