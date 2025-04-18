# Licenciamento e Analytics Tab
with config_tab7:
    st.markdown("### Licenciamento e Telemetria")
    
    st.info("""
    Configure sua licença e opções de telemetria nesta seção. 
    O envio de dados de performance é necessário para o modelo de compartilhamento de lucros.
    """)
    
    # Create license and telemetry settings UI
    license_col1, license_col2 = st.columns(2)
    
    with license_col1:
        # User ID (read-only if already exists)
        if 'performance_tracker' in st.session_state and st.session_state.performance_tracker and st.session_state.performance_tracker.user_id:
            user_id = st.text_input(
                "ID do Usuário",
                value=st.session_state.performance_tracker.user_id,
                disabled=True,
                help="Identificador único do usuário (gerado automaticamente)"
            )
        else:
            user_id = st.text_input(
                "ID do Usuário (opcional)",
                value="",
                help="Identificador único do usuário (deixe em branco para gerar automaticamente)"
            )
    
    with license_col2:
        # License key
        if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
            current_license = st.session_state.performance_tracker.license_key or ""
        else:
            current_license = os.environ.get("LICENSE_KEY", "")
            
        license_key = st.text_input(
            "Chave de Licença",
            value=current_license,
            help="Chave de licença para ativar recursos premium"
        )
    
    # API Key for telemetry
    if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
        current_api_key = st.session_state.performance_tracker.api_key or ""
    else:
        current_api_key = os.environ.get("ANALYTICS_API_KEY", "")
        
    api_key = st.text_input(
        "Chave de API para Telemetria",
        value=current_api_key,
        type="password",
        help="Chave para autenticação com o servidor de telemetria"
    )
    
    # Commission settings
    st.markdown("#### Configurações de Comissão")
    
    # Get current commission rate
    if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
        current_rate = st.session_state.performance_tracker.commission_rate
    else:
        current_rate = 0.20  # Default 20%
    
    commission_rate = st.slider(
        "Taxa de Comissão",
        min_value=0.0,
        max_value=0.5,
        value=current_rate,
        step=0.01,
        format="%.2f",
        help="Porcentagem do lucro devida como comissão (modelo de compartilhamento de lucros)"
    )
    
    # Telemetry settings
    st.markdown("#### Configurações de Telemetria")
    
    # Get current send interval
    if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
        current_interval = st.session_state.performance_tracker.send_interval
    else:
        current_interval = 24  # Default 24 hours
    
    send_interval = st.number_input(
        "Intervalo de Envio (horas)",
        min_value=1,
        max_value=168,  # Up to 7 days
        value=current_interval,
        help="Intervalo em horas para envio automático de dados de performance"
    )
    
    # Enable/disable telemetry
    enable_telemetry = st.toggle(
        "Ativar Telemetria",
        value=True,
        help="Habilita o envio de dados de performance para o servidor"
    )
    
    # Save license and telemetry settings button
    if st.button("Salvar Configurações de Licenciamento"):
        try:
            # Update environment variables
            if user_id:
                os.environ["USER_ID"] = user_id
            
            if license_key:
                os.environ["LICENSE_KEY"] = license_key
            
            if api_key:
                os.environ["ANALYTICS_API_KEY"] = api_key
            
            # If tracker exists, update it
            if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
                if user_id:
                    st.session_state.performance_tracker.user_id = user_id
                
                if license_key:
                    st.session_state.performance_tracker.license_key = license_key
                
                if api_key:
                    st.session_state.performance_tracker.api_key = api_key
                
                # Update commission rate
                st.session_state.performance_tracker.update_commission_rate(commission_rate)
                
                # Update send interval
                st.session_state.performance_tracker.send_interval = send_interval
                
                # Update config
                update_config_file({
                    "user_id": user_id if user_id else st.session_state.performance_tracker.user_id,
                    "license_key": license_key,
                    "analytics_api_key": api_key,
                    "commission_rate": commission_rate,
                    "telemetry_interval": send_interval,
                    "telemetry_enabled": enable_telemetry
                })
                
                st.success("Configurações de licenciamento salvas com sucesso!")
            else:
                # Initialize tracker with new settings
                from user_analytics import initialize_analytics
                
                performance_tracker, usage_collector = initialize_analytics(
                    user_id=user_id,
                    license_key=license_key,
                    api_key=api_key
                )
                
                # Update settings
                if performance_tracker:
                    performance_tracker.update_commission_rate(commission_rate)
                    performance_tracker.send_interval = send_interval
                    
                    # Store in session state
                    st.session_state.performance_tracker = performance_tracker
                    st.session_state.usage_collector = usage_collector
                    
                    # Update config
                    update_config_file({
                        "user_id": user_id if user_id else performance_tracker.user_id,
                        "license_key": license_key,
                        "analytics_api_key": api_key,
                        "commission_rate": commission_rate,
                        "telemetry_interval": send_interval,
                        "telemetry_enabled": enable_telemetry
                    })
                    
                    st.success("Configurações de licenciamento salvas com sucesso!")
                else:
                    st.error("Falha ao inicializar telemetria. Verifique as configurações.")
        except Exception as e:
            st.error(f"Erro ao salvar configurações: {str(e)}")
    
    # Performance data and reports
    st.markdown("#### Dados de Performance")
    
    if 'performance_tracker' in st.session_state and st.session_state.performance_tracker:
        tracker = st.session_state.performance_tracker
        
        # Display commission due
        commission_summary = tracker.get_commission_summary()
        
        st.metric(
            label="Lucro Total Registrado",
            value=f"R$ {commission_summary['total_profit']:,.2f}"
        )
        
        st.metric(
            label="Comissão Devida",
            value=f"R$ {commission_summary['commission_due']:,.2f}",
            delta=f"{commission_summary['commission_rate']:.2%} do lucro"
        )
        
        # Period selector for reports
        report_period = st.selectbox(
            "Período para Relatório",
            options=["7 dias", "14 dias", "30 dias", "60 dias", "90 dias"],
            index=2  # Default to 30 days
        )
        
        # Convert period to days
        if report_period == "7 dias":
            days = 7
        elif report_period == "14 dias":
            days = 14
        elif report_period == "30 dias":
            days = 30
        elif report_period == "60 dias":
            days = 60
        else:
            days = 90
        
        # Generate report button
        report_col1, report_col2, report_col3 = st.columns(3)
        
        with report_col1:
            if st.button("Gerar Relatório de Performance", use_container_width=True):
                with st.spinner("Gerando relatório..."):
                    try:
                        # Generate report with equity curve
                        report = tracker.generate_performance_report(
                            days=days, 
                            include_equity_curve=True
                        )
                        
                        if report:
                            # Store in session state
                            st.session_state.performance_report = report
                            
                            # Display success message
                            st.success("Relatório gerado com sucesso!")
                            
                            # Update usage statistics
                            if 'usage_collector' in st.session_state and st.session_state.usage_collector:
                                st.session_state.usage_collector.record_feature_usage("generate_report")
                    except Exception as e:
                        st.error(f"Erro ao gerar relatório: {str(e)}")
        
        with report_col2:
            if st.button("Salvar Relatório (JSON)", use_container_width=True):
                try:
                    # Save report to file
                    filename = f"relatorio_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    success = tracker.save_report(filename=filename, format="json")
                    
                    if success:
                        st.success(f"Relatório salvo como: {filename}")
                    else:
                        st.error("Falha ao salvar relatório")
                except Exception as e:
                    st.error(f"Erro ao salvar relatório: {str(e)}")
        
        with report_col3:
            if st.button("Salvar Relatório (HTML)", use_container_width=True):
                try:
                    # Save report to file
                    filename = f"relatorio_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    success = tracker.save_report(filename=filename, format="html")
                    
                    if success:
                        st.success(f"Relatório salvo como: {filename}")
                    else:
                        st.error("Falha ao salvar relatório")
                except Exception as e:
                    st.error(f"Erro ao salvar relatório: {str(e)}")
        
        # Manual telemetry actions
        st.markdown("#### Ações de Telemetria")
        
        telemetry_col1, telemetry_col2 = st.columns(2)
        
        with telemetry_col1:
            if st.button("Enviar Dados Agora", use_container_width=True):
                with st.spinner("Enviando dados para o servidor..."):
                    try:
                        # Force send data
                        success = tracker.send_performance_data(force=True)
                        
                        if success:
                            st.success("Dados enviados com sucesso!")
                        else:
                            st.warning("Sem dados para enviar ou falha no envio.")
                    except Exception as e:
                        st.error(f"Erro ao enviar dados: {str(e)}")
        
        with telemetry_col2:
            if st.button("Verificar Status de Licença", use_container_width=True):
                with st.spinner("Verificando status da licença..."):
                    try:
                        # Send data (which also checks license)
                        success = tracker.send_performance_data(force=True)
                        
                        if success:
                            if tracker.license_key:
                                st.success(f"Licença ativa: {tracker.license_key}")
                            else:
                                st.warning("Nenhuma licença configurada")
                        else:
                            st.error("Falha ao verificar licença. Servidor indisponível.")
                    except Exception as e:
                        st.error(f"Erro ao verificar licença: {str(e)}")
        
        # Display performance summary if report exists
        if 'performance_report' in st.session_state:
            report = st.session_state.performance_report
            
            st.markdown("#### Resumo de Performance")
            
            # Create metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="Trades Totais",
                    value=report["performance"]["trades"]
                )
            
            with metric_col2:
                st.metric(
                    label="Taxa de Acerto",
                    value=f"{report['performance']['win_rate']:.2%}"
                )
            
            with metric_col3:
                st.metric(
                    label="Resultado Líquido",
                    value=f"R$ {report['performance']['net_pnl']:,.2f}"
                )
            
            with metric_col4:
                st.metric(
                    label="Fator de Lucro",
                    value=f"{report['performance']['profit_factor']:.2f}"
                )
            
            # Display equity curve if available
            if "equity_curve" in report:
                st.markdown("#### Curva de Patrimônio")
                st.image(f"data:image/png;base64,{report['equity_curve']}")
    else:
        st.warning("""
        Telemetria não está inicializada. Salve as configurações 
        de licenciamento para ativar o rastreamento de performance.
        """)