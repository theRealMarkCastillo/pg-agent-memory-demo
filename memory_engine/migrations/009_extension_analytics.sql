-- A distinct login per tenant prevents custom-GUC tenant impersonation. RLS
-- uses session_user (not current_user, which SET ROLE can change).
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='agent_ext_reader') THEN
        CREATE ROLE agent_ext_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    END IF;
END $$;
CREATE TABLE ext_analytics_identities(role_name name PRIMARY KEY,user_id text NOT NULL);
REVOKE ALL ON ext_analytics_identities FROM PUBLIC;
CREATE FUNCTION ext_analytics_tenant() RETURNS text
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog
AS $$ SELECT user_id FROM public.ext_analytics_identities WHERE role_name=session_user $$;
REVOKE ALL ON FUNCTION ext_analytics_tenant() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION ext_analytics_tenant() TO agent_ext_reader;
CREATE TABLE ext_customers (
    user_id text NOT NULL, customer_id integer NOT NULL, region text NOT NULL,
    PRIMARY KEY(user_id,customer_id)
);
CREATE TABLE ext_orders (
    user_id text NOT NULL, order_id integer NOT NULL, customer_id integer NOT NULL,
    amount integer NOT NULL CHECK(amount >= 0), status text NOT NULL CHECK(status IN ('open','paid')),
    internal_note text NOT NULL DEFAULT '',
    PRIMARY KEY(user_id,order_id),
    FOREIGN KEY(user_id,customer_id) REFERENCES ext_customers
);
ALTER TABLE ext_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE ext_orders FORCE ROW LEVEL SECURITY;
ALTER TABLE ext_customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE ext_customers FORCE ROW LEVEL SECURITY;
CREATE POLICY ext_order_tenant ON ext_orders TO agent_ext_reader
    USING(user_id=public.ext_analytics_tenant());
CREATE POLICY ext_customer_tenant ON ext_customers TO agent_ext_reader
    USING(user_id=public.ext_analytics_tenant());
GRANT USAGE ON SCHEMA public TO agent_ext_reader;
GRANT SELECT(user_id,order_id,customer_id,amount,status) ON ext_orders TO agent_ext_reader;
GRANT SELECT ON ext_customers TO agent_ext_reader;
CREATE VIEW ext_order_metrics WITH(security_invoker=true,security_barrier=true)
AS SELECT order_id,customer_id,amount,status FROM ext_orders;
CREATE VIEW ext_order_regions WITH(security_invoker=true,security_barrier=true)
AS SELECT o.order_id,o.amount,o.status,c.region FROM ext_orders o
JOIN ext_customers c ON o.user_id=c.user_id AND o.customer_id=c.customer_id;
GRANT SELECT ON ext_order_metrics,ext_order_regions TO agent_ext_reader;
CREATE TABLE ext_analytics_runs (
    run_id uuid PRIMARY KEY, user_id text NOT NULL, plan jsonb NOT NULL,
    result jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
